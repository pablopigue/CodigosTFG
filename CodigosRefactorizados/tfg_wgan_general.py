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

# Opciones válidas: "MNIST", "FashionMNIST", "SVHN"
DATASET_NAME = "MNIST"
MODEL_NAME = "WGAN"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hiperparámetros estándar WGAN. Referencia: Arjovsky et al. (2017), arXiv:1701.07875
LR = 0.00005
BATCH_SIZE = 64
Z_DIM = 100
SAVE_IMG_FREQ = 5
CALC_METRICS_FREQ = 5
NUM_RUNS = 10
NUM_EVAL_IMAGES = 10000
IMG_SIZE = 32

CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

if DATASET_NAME in ["MNIST", "FashionMNIST"]:
    CHANNELS = 1
    EPOCHS = 80
elif DATASET_NAME == "SVHN":
    CHANNELS = 3
    EPOCHS = 150
else:
    raise ValueError(f"Dataset '{DATASET_NAME}' no reconocido. Opciones válidas: MNIST, FashionMNIST, SVHN.")

IMG_DIM = CHANNELS * IMG_SIZE * IMG_SIZE

EXPERIMENT_DIR = f"/mnt/homeGPU/pablomarpa/CodigosTFG/tfg_wganRF_{DATASET_NAME.lower()}"
make_experiment_dirs(EXPERIMENT_DIR)

print(f"Iniciando entrenamiento WGAN en: {DEVICE} con dataset {DATASET_NAME}. Ejecuciones totales: {NUM_RUNS}", flush=True)

# ==========================================
# 2. CARGA DE DATOS
# ==========================================
dataset = load_dataset(DATASET_NAME, IMG_SIZE, CHANNELS)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ==========================================
# 3. SUBCONJUNTO FIJO DE EVALUACIÓN PARA FID
# ==========================================
real_eval_images = build_fixed_eval_set(
    dataset, BATCH_SIZE, NUM_EVAL_IMAGES, DEVICE, CHANNELS
)

# ==========================================
# 4. DEFINICIÓN DE CLASES
# ==========================================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(IMG_DIM, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
            # Sin Sigmoid: salida escalar sin acotar
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(Z_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, IMG_DIM),
            nn.Tanh()
        )
    def forward(self, z):
        return self.gen(z)

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
            real = data[0].view(-1, IMG_DIM).to(DEVICE)
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
                flatten_output=True
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
                title=f"Imágenes Generadas WGAN ({DATASET_NAME}) - Época {epoch+1}",
                flatten_output=True
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

print(f"Entrenamiento WGAN de {NUM_RUNS} ejecuciones finalizado. Todo guardado en: {EXPERIMENT_DIR}", flush=True)
