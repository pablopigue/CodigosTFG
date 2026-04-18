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

# Opciones válidas: "MNIST", "FashionMNIST", "SVHN"
DATASET_NAME = "MNIST"
MODEL_NAME = "DCGAN+LS"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0002
BETA1 = 0.5   # Estándar DCGAN. Referencia: Radford et al. (2015), arXiv:1511.06434
BATCH_SIZE = 128
Z_DIM = 100
SAVE_IMG_FREQ = 5
CALC_METRICS_FREQ = 5
NUM_RUNS = 10
NUM_EVAL_IMAGES = 10000
IMG_SIZE = 32
# Label smoothing: etiquetas reales a 0.9 en lugar de 1.0 solo para el discriminador.
# El generador sigue usando 1.0. Única diferencia respecto a DCGAN base.
REAL_LABEL_VALUE = 0.9

if DATASET_NAME in ["MNIST", "FashionMNIST"]:
    CHANNELS = 1
    EPOCHS = 80
elif DATASET_NAME == "SVHN":
    CHANNELS = 3
    EPOCHS = 150
else:
    raise ValueError(f"Dataset '{DATASET_NAME}' no reconocido. Opciones válidas: MNIST, FashionMNIST, SVHN.")

EXPERIMENT_DIR = f"/mnt/homeGPU/pablomarpa/CodigosTFG/tfg_dcganRF_ls_{DATASET_NAME.lower()}"
make_experiment_dirs(EXPERIMENT_DIR)

print(f"Iniciando entrenamiento DCGAN+LS en: {DEVICE} con dataset {DATASET_NAME}. Ejecuciones totales: {NUM_RUNS}", flush=True)
print(f"Label smoothing activo: etiquetas reales = {REAL_LABEL_VALUE}", flush=True)

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
# Arquitectura idéntica a DCGAN base.
# ==========================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 256, 4, 1, 0, bias=False),
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
    disc = Discriminator().to(DEVICE)
    gen.apply(weights_init_dcgan)
    disc.apply(weights_init_dcgan)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(BETA1, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM).to(DEVICE)

    history = {"epoch": [], "loss_g": [], "loss_d": [], "fid": [], "is_mean": [], "is_std": []}

    for epoch in range(EPOCHS):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        gen.train()
        disc.train()
        for batch_idx, data in enumerate(loader):
            real = data[0].to(DEVICE)
            batch_size_curr = real.shape[0]

            disc.zero_grad()
            # Label smoothing: reales a 0.9, falsas a 0.0
            label_real = torch.full((batch_size_curr,), REAL_LABEL_VALUE, device=DEVICE)
            output_real = disc(real)
            loss_disc_real = criterion(output_real, label_real)
            loss_disc_real.backward()

            noise = torch.randn(batch_size_curr, Z_DIM).to(DEVICE)
            fake = gen(noise)
            output_fake = disc(fake.detach())
            loss_disc_fake = criterion(output_fake, torch.zeros(batch_size_curr, device=DEVICE))
            loss_disc_fake.backward()
            opt_disc.step()
            epoch_loss_d += (loss_disc_real + loss_disc_fake).item()

            # Generador apunta a 1.0, no usa smoothing
            gen.zero_grad()
            output_gen = disc(fake)
            loss_gen = criterion(output_gen, torch.ones(batch_size_curr, device=DEVICE))
            loss_gen.backward()
            opt_gen.step()
            epoch_loss_g += loss_gen.item()

        avg_loss_g = epoch_loss_g / len(loader)
        avg_loss_d = epoch_loss_d / len(loader)

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
            losses={'D': avg_loss_d, 'G': avg_loss_g},
            metrics={'fid': current_fid, 'is_mean': current_is_mean, 'is_std': current_is_std}
        ), flush=True)

        history["epoch"].append(epoch + 1)
        history["loss_g"].append(avg_loss_g)
        history["loss_d"].append(avg_loss_d)
        history["fid"].append(current_fid)
        history["is_mean"].append(current_is_mean)
        history["is_std"].append(current_is_std)

        if run == 0 and ((epoch + 1) % SAVE_IMG_FREQ == 0 or epoch == 0):
            save_sample_images(
                gen, fixed_noise, CHANNELS, IMG_SIZE,
                filepath=f"{EXPERIMENT_DIR}/images/epoch_{epoch+1}.png",
                title=f"Imágenes Generadas DCGAN+LS ({DATASET_NAME}) - Época {epoch+1}",
                flatten_output=False
            )

    df_run = pd.DataFrame(history)
    df_run['run'] = run + 1
    all_runs_data.append(df_run)

    if run == 0:
        save_run_artifacts(
            gen, disc, df_run, EXPERIMENT_DIR,
            model_name=MODEL_NAME, dataset_name=DATASET_NAME,
            critic_name='discriminator'
        )

# ==========================================
# 6. POST-PROCESADO Y PROMEDIADO FINAL
# ==========================================
print("\nGenerando gráficas promediadas...", flush=True)
aggregate_runs(all_runs_data, EXPERIMENT_DIR, MODEL_NAME, DATASET_NAME)

print(f"Entrenamiento DCGAN+LS de {NUM_RUNS} ejecuciones finalizado. Todo guardado en: {EXPERIMENT_DIR}", flush=True)
