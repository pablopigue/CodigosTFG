import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import (
    load_dataset,
    build_fixed_eval_set,
    compute_fid_is,
    weights_init_dcgan,
    save_sample_images,
    save_run_artifacts,
    aggregate_runs,
    make_experiment_dirs,
    format_epoch_log,
)

# ==========================================
# 1. CONFIGURACIÓN GENERAL Y DEL DATASET
# ==========================================
plt.switch_backend('agg')

DATASET_NAME = "CelebA"
MODEL_NAME = "WGAN-Conv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hiperparámetros estándar WGAN. Referencia: Arjovsky et al. (2017), arXiv:1701.07875
# Arquitectura convolucional basada en DCGAN (Radford et al., 2015) con pérdida Wasserstein.
LR = 0.00005
BATCH_SIZE = 64
Z_DIM = 100
SAVE_IMG_FREQ = 5
CALC_METRICS_FREQ = 5
NUM_RUNS = 5
NUM_EVAL_IMAGES = 10000
IMG_SIZE = 64
CHANNELS = 3
EPOCHS = 40

CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

EXPERIMENT_DIR = f"/mnt/homeGPU/pablomarpa/CodigosTFG/tfg_wganRF_conv_{DATASET_NAME.lower()}"
DATA_DIR = "/mnt/homeGPU/pablomarpa/data/celeba"
make_experiment_dirs(EXPERIMENT_DIR)

print(f"Iniciando entrenamiento WGAN-Conv en: {DEVICE} con dataset {DATASET_NAME}. Ejecuciones totales: {NUM_RUNS}", flush=True)

# ==========================================
# 2. CARGA DE DATOS
# ==========================================
dataset = load_dataset(DATASET_NAME, IMG_SIZE, CHANNELS, data_dir=DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ==========================================
# 3. SUBCONJUNTO FIJO DE EVALUACIÓN PARA FID
# ==========================================
real_eval_images = build_fixed_eval_set(
    dataset, BATCH_SIZE, NUM_EVAL_IMAGES, DEVICE, CHANNELS
)

# ==========================================
# 4. DEFINICIÓN DE CLASES
# Arquitectura convolucional idéntica a DCGAN, adaptada a 64x64.
# Cambios respecto a DCGAN:
#   - Crítico sin Sigmoid (salida escalar sin acotar)
#   - Pérdida Wasserstein en lugar de BCE
#   - Weight clipping para restricción 1-Lipschitz
#   - RMSprop en lugar de Adam
# ==========================================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
        self.main = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Sin Sigmoid: salida escalar sin acotar
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.main(x).view(-1)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.main(z.view(-1, Z_DIM, 1, 1))

# ==========================================
# 5. BUCLE MULTI-RUN
# ==========================================
all_runs_data = []

for run in range(NUM_RUNS):
    print(f"\n{'='*40}")
    print(f" INICIANDO EJECUCIÓN {run + 1} DE {NUM_RUNS}")
    print(f"{'='*40}\n", flush=True)

    gen = Generator().to(DEVICE)
    critic = Critic().to(DEVICE)
    gen.apply(weights_init_dcgan)
    critic.apply(weights_init_dcgan)

    # RMSprop sin momentum, tal como especifica el paper original de WGAN.
    opt_gen = optim.RMSprop(gen.parameters(), lr=LR)
    opt_critic = optim.RMSprop(critic.parameters(), lr=LR)

    fixed_noise = torch.randn(32, Z_DIM).to(DEVICE)

    history = {"epoch": [], "loss_g": [], "loss_c": [], "fid": [], "is_mean": [], "is_std": []}

    for epoch in range(EPOCHS):
        epoch_loss_g = 0.0
        epoch_loss_c = 0.0

        gen.train()
        critic.train()
        for batch_idx, data in enumerate(loader):
            real = data[0].to(DEVICE)
            batch_size_curr = real.shape[0]

            # 1. ENTRENAR CRÍTICO (CRITIC_ITERATIONS veces consecutivas)
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(batch_size_curr, Z_DIM).to(DEVICE)
                fake = gen(noise)

                critic_real = critic(real).view(-1)
                critic_fake = critic(fake.detach()).view(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            epoch_loss_c += loss_critic.item()

            # 2. ENTRENAR GENERADOR (1 vez, con ruido nuevo)
            noise = torch.randn(batch_size_curr, Z_DIM).to(DEVICE)
            fake_for_gen = gen(noise)
            loss_gen = -torch.mean(critic(fake_for_gen).view(-1))

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            epoch_loss_g += loss_gen.item()

        avg_loss_c = epoch_loss_c / len(loader)
        avg_loss_g = epoch_loss_g / len(loader)

        current_fid = np.nan
        current_is_mean = np.nan
        current_is_std = np.nan

        if (epoch + 1) % CALC_METRICS_FREQ == 0 or epoch == 0 or epoch == EPOCHS - 1:
            gen.eval()
            current_fid, current_is_mean, current_is_std = compute_fid_is(
                gen, real_eval_images, Z_DIM, DEVICE, CHANNELS, IMG_SIZE,
                flatten_output=False
            )

        print(format_epoch_log(
            run + 1, NUM_RUNS, epoch + 1, EPOCHS,
            losses={'C': avg_loss_c, 'G': avg_loss_g},
            metrics={'fid': current_fid, 'is_mean': current_is_mean, 'is_std': current_is_std}
        ), flush=True)

        history["epoch"].append(epoch + 1)
        history["loss_g"].append(avg_loss_g)
        history["loss_c"].append(avg_loss_c)
        history["fid"].append(current_fid)
        history["is_mean"].append(current_is_mean)
        history["is_std"].append(current_is_std)

        if run == 0 and ((epoch + 1) % SAVE_IMG_FREQ == 0 or epoch == 0):
            save_sample_images(
                gen, fixed_noise, CHANNELS, IMG_SIZE,
                filepath=f"{EXPERIMENT_DIR}/images/epoch_{epoch+1}.png",
                title=f"Imágenes Generadas WGAN-Conv ({DATASET_NAME}) - Época {epoch+1}",
                flatten_output=False
            )

    df_run = pd.DataFrame(history)
    df_run['run'] = run + 1
    all_runs_data.append(df_run)

    if run == 0:
        save_run_artifacts(
            gen, critic, df_run, EXPERIMENT_DIR,
            model_name=MODEL_NAME, dataset_name=DATASET_NAME,
            critic_name='critic'
        )

# ==========================================
# 6. POST-PROCESADO Y PROMEDIADO FINAL
# ==========================================
print("\nGenerando gráficas promediadas...", flush=True)
aggregate_runs(all_runs_data, EXPERIMENT_DIR, MODEL_NAME, DATASET_NAME)

print(f"Entrenamiento WGAN-Conv de {NUM_RUNS} ejecuciones finalizado. Todo guardado en: {EXPERIMENT_DIR}", flush=True)
