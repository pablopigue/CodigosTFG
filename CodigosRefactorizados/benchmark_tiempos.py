"""
benchmark_tiempos.py

Mide el tiempo puro de entrenamiento por época de cada combinación
modelo × dataset. No calcula FID/IS ni guarda imágenes, para obtener
una medida limpia del coste computacional del entrenamiento adversario.

Metodología:
    - 1 época de warmup por combinación (NO se cronometra): permite que
      la GPU compile kernels, llene caches y estabilice la medición.
    - 5 épocas cronometradas por combinación: se reporta la media y
      la desviación estándar.
    - torch.cuda.synchronize() antes y después de cada época garantiza
      mediciones precisas dado el carácter asíncrono de CUDA.

Uso:
    python benchmark_tiempos.py

Salida:
    - benchmark_results.csv: tabla (dataset, model, mean_time, std_time, ...)
    - benchmark_results.json: misma información en formato legible.

Tiempo estimado total: 1-2 horas en una GTX Titan Xp.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from utils import load_dataset

# ============================================================
# CONFIGURACIÓN
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}", flush=True)

WARMUP_EPOCHS = 1
MEASURED_EPOCHS = 5

Z_DIM = 100
DATA_DIR_CELEBA = "/mnt/homeGPU/pablomarpa/data/celeba"

# Datasets a evaluar
DATASETS = [
    # (dataset_name, img_size, channels, data_dir_o_None)
    ("MNIST",        32, 1, None),
    ("FashionMNIST", 32, 1, None),
    ("SVHN",         32, 3, None),
    ("CelebA",       64, 3, DATA_DIR_CELEBA),
]

# Hiperparámetros específicos de cada modelo (tal como en los scripts)
MODEL_CONFIGS = {
    "Vanilla GAN": {
        "arch": "MLP", "loss": "BCE",
        "lr": 0.0002, "batch_size": 128, "critic_iters": 1,
        "optimizer": "Adam", "adam_betas": (0.5, 0.999),
    },
    "DCGAN": {
        "arch": "CNN", "loss": "BCE",
        "lr": 0.0002, "batch_size": 128, "critic_iters": 1,
        "optimizer": "Adam", "adam_betas": (0.5, 0.999),
    },
    "DCGAN+LS": {
        "arch": "CNN", "loss": "BCE",
        "lr": 0.0002, "batch_size": 128, "critic_iters": 1,
        "optimizer": "Adam", "adam_betas": (0.5, 0.999),
        "label_smoothing": 0.9,
    },
    "WGAN": {
        "arch": "MLP", "loss": "Wasserstein",
        "lr": 0.00005, "batch_size": 64, "critic_iters": 5,
        "optimizer": "RMSprop", "weight_clip": 0.01,
    },
    "WGAN-Conv": {
        "arch": "CNN", "loss": "Wasserstein",
        "lr": 0.00005, "batch_size": 64, "critic_iters": 5,
        "optimizer": "RMSprop", "weight_clip": 0.01,
    },
    "WGAN-GP": {
        "arch": "CNN", "loss": "Wasserstein-GP",
        "lr": 0.0001, "batch_size": 64, "critic_iters": 5,
        "optimizer": "Adam", "adam_betas": (0.0, 0.9),
        "lambda_gp": 10,
    },
}

# ============================================================
# ARQUITECTURAS
# Reproducen exactamente las de los scripts de entrenamiento.
# ============================================================

# MLP (Vanilla GAN, WGAN)
class GeneratorMLP(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, img_dim), nn.Tanh()
        )
    def forward(self, z):
        return self.gen(z)

class DiscriminatorMLP(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)

class CriticMLP(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.disc(x)


# CNN 32x32
class GeneratorCNN32(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.main(z.view(-1, Z_DIM, 1, 1))

class DiscriminatorCNN32(nn.Module):
    """DCGAN, DCGAN+LS, WGAN-Conv (este último con use_sigmoid=False)."""
    def __init__(self, channels, use_sigmoid=True):
        super().__init__()
        layers = [
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x).view(-1)

class CriticCNN32GP(nn.Module):
    """WGAN-GP con GroupNorm."""
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(8, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(8, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.main(x).view(-1)


# CNN 64x64 (CelebA)
class GeneratorCNN64(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.main(z.view(-1, Z_DIM, 1, 1))

class DiscriminatorCNN64(nn.Module):
    """DCGAN, DCGAN+LS, WGAN-Conv en CelebA (este último con use_sigmoid=False)."""
    def __init__(self, channels, use_sigmoid=True):
        super().__init__()
        layers = [
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x).view(-1)

class CriticCNN64GP(nn.Module):
    """WGAN-GP en CelebA con GroupNorm."""
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(8, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(8, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.GroupNorm(8, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.main(x).view(-1)


# ============================================================
# FACTORY DE MODELOS Y OPTIMIZADORES
# ============================================================

def build_networks(model_name, img_size, channels):
    """Devuelve (gen, disc_o_critic) ya en DEVICE según modelo y resolución."""
    cfg = MODEL_CONFIGS[model_name]
    img_dim = channels * img_size * img_size

    if cfg["arch"] == "MLP":
        gen = GeneratorMLP(img_dim).to(DEVICE)
        if model_name == "Vanilla GAN":
            net = DiscriminatorMLP(img_dim).to(DEVICE)
        else:  # WGAN
            net = CriticMLP(img_dim).to(DEVICE)
        return gen, net

    # CNN
    if img_size == 32:
        gen = GeneratorCNN32(channels).to(DEVICE)
        if model_name == "WGAN-GP":
            net = CriticCNN32GP(channels).to(DEVICE)
        else:
            use_sigmoid = (cfg["loss"] == "BCE")
            net = DiscriminatorCNN32(channels, use_sigmoid=use_sigmoid).to(DEVICE)
    else:  # img_size == 64
        gen = GeneratorCNN64(channels).to(DEVICE)
        if model_name == "WGAN-GP":
            net = CriticCNN64GP(channels).to(DEVICE)
        else:
            use_sigmoid = (cfg["loss"] == "BCE")
            net = DiscriminatorCNN64(channels, use_sigmoid=use_sigmoid).to(DEVICE)
    return gen, net


def build_optimizers(model_name, gen, disc_or_critic):
    """Devuelve (opt_gen, opt_disc_or_critic) según la configuración."""
    cfg = MODEL_CONFIGS[model_name]
    if cfg["optimizer"] == "Adam":
        opt_gen = optim.Adam(gen.parameters(), lr=cfg["lr"],
                             betas=cfg["adam_betas"])
        opt_net = optim.Adam(disc_or_critic.parameters(), lr=cfg["lr"],
                             betas=cfg["adam_betas"])
    else:  # RMSprop
        opt_gen = optim.RMSprop(gen.parameters(), lr=cfg["lr"])
        opt_net = optim.RMSprop(disc_or_critic.parameters(), lr=cfg["lr"])
    return opt_gen, opt_net


# ============================================================
# PASOS DE ENTRENAMIENTO
# Cada función toma un batch y ejecuta una actualización completa
# (critic_iters pasos de crítico + 1 de generador), tal como en los scripts.
# ============================================================

def step_vanilla_or_dcgan(model_name, real, gen, disc, opt_gen, opt_disc,
                          criterion, img_dim=None):
    """Vanilla GAN, DCGAN, DCGAN+LS. Un paso por batch."""
    cfg = MODEL_CONFIGS[model_name]
    batch_size_curr = real.shape[0]

    # Reshape para MLP
    if cfg["arch"] == "MLP":
        real = real.view(-1, img_dim)

    # --- Discriminador ---
    disc.zero_grad()
    output_real = disc(real).view(-1)
    label_real_value = cfg.get("label_smoothing", 1.0)
    label_real = torch.full((batch_size_curr,), label_real_value, device=DEVICE)
    loss_d_real = criterion(output_real, label_real)
    loss_d_real.backward()

    noise = torch.randn(batch_size_curr, Z_DIM, device=DEVICE)
    fake = gen(noise)
    output_fake = disc(fake.detach()).view(-1)
    loss_d_fake = criterion(output_fake,
                            torch.zeros(batch_size_curr, device=DEVICE))
    loss_d_fake.backward()
    opt_disc.step()

    # --- Generador ---
    gen.zero_grad()
    output_gen = disc(fake).view(-1)
    loss_g = criterion(output_gen,
                       torch.ones(batch_size_curr, device=DEVICE))
    loss_g.backward()
    opt_gen.step()


def step_wgan(model_name, real, gen, critic, opt_gen, opt_critic, img_dim=None):
    """WGAN y WGAN-Conv: 5 pasos de crítico (con weight clipping) + 1 de generador."""
    cfg = MODEL_CONFIGS[model_name]
    batch_size_curr = real.shape[0]

    if cfg["arch"] == "MLP":
        real = real.view(-1, img_dim)

    for _ in range(cfg["critic_iters"]):
        noise = torch.randn(batch_size_curr, Z_DIM, device=DEVICE)
        fake = gen(noise)
        critic_real = critic(real).view(-1)
        critic_fake = critic(fake.detach()).view(-1)
        loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake))

        critic.zero_grad()
        loss_c.backward()
        opt_critic.step()

        # Weight clipping
        for p in critic.parameters():
            p.data.clamp_(-cfg["weight_clip"], cfg["weight_clip"])

    noise = torch.randn(batch_size_curr, Z_DIM, device=DEVICE)
    fake_for_gen = gen(noise)
    loss_g = -torch.mean(critic(fake_for_gen).view(-1))
    gen.zero_grad()
    loss_g.backward()
    opt_gen.step()


def compute_gradient_penalty(critic, real, fake):
    """GP estándar de WGAN-GP."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=DEVICE)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def step_wgan_gp(real, gen, critic, opt_gen, opt_critic):
    """WGAN-GP: 5 pasos de crítico (con GP) + 1 de generador."""
    cfg = MODEL_CONFIGS["WGAN-GP"]
    batch_size_curr = real.shape[0]

    for _ in range(cfg["critic_iters"]):
        noise = torch.randn(batch_size_curr, Z_DIM, device=DEVICE)
        fake = gen(noise)
        critic_real = critic(real).view(-1)
        critic_fake = critic(fake.detach()).view(-1)
        gp = compute_gradient_penalty(critic, real, fake.detach())
        loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake)) \
                 + cfg["lambda_gp"] * gp

        critic.zero_grad()
        loss_c.backward()
        opt_critic.step()

    noise = torch.randn(batch_size_curr, Z_DIM, device=DEVICE)
    fake_for_gen = gen(noise)
    loss_g = -torch.mean(critic(fake_for_gen).view(-1))
    gen.zero_grad()
    loss_g.backward()
    opt_gen.step()


# ============================================================
# EJECUCIÓN DE UNA COMBINACIÓN MODELO × DATASET
# ============================================================

def run_one_combo(model_name, dataset_name, img_size, channels, data_dir):
    """Entrena warmup+measured épocas y devuelve lista de tiempos medidos."""
    cfg = MODEL_CONFIGS[model_name]

    print(f"\n{'='*60}")
    print(f"  {model_name}  /  {dataset_name}  ({img_size}x{img_size}, "
          f"{channels} ch)")
    print(f"{'='*60}", flush=True)

    # --- Dataset y loader ---
    dataset = load_dataset(dataset_name, img_size, channels,
                           data_dir=data_dir)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        shuffle=True, drop_last=True, num_workers=2)
    print(f"  Batches por época: {len(loader)} "
          f"(batch_size={cfg['batch_size']})", flush=True)

    # --- Redes y optimizadores ---
    gen, net = build_networks(model_name, img_size, channels)
    opt_gen, opt_net = build_optimizers(model_name, gen, net)
    img_dim = channels * img_size * img_size if cfg["arch"] == "MLP" else None
    criterion = nn.BCELoss() if cfg["loss"] == "BCE" else None

    gen.train()
    net.train()

    epoch_times = []

    for epoch in range(WARMUP_EPOCHS + MEASURED_EPOCHS):
        # Sincronizar GPU antes de medir
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for data in loader:
            real = data[0].to(DEVICE)
            if cfg["loss"] == "BCE":
                step_vanilla_or_dcgan(model_name, real, gen, net,
                                      opt_gen, opt_net, criterion,
                                      img_dim=img_dim)
            elif cfg["loss"] == "Wasserstein":
                step_wgan(model_name, real, gen, net,
                          opt_gen, opt_net, img_dim=img_dim)
            else:  # Wasserstein-GP
                step_wgan_gp(real, gen, net, opt_gen, opt_net)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if epoch < WARMUP_EPOCHS:
            print(f"  [warmup] Época {epoch+1}: {elapsed:.2f}s", flush=True)
        else:
            epoch_times.append(elapsed)
            print(f"  [medida] Época {epoch+1}: {elapsed:.2f}s", flush=True)

    mean_t = float(np.mean(epoch_times))
    std_t = float(np.std(epoch_times, ddof=1)) if len(epoch_times) > 1 else 0.0
    total_batches = len(loader)
    print(f"  --> Media: {mean_t:.2f}s ± {std_t:.2f}s "
          f"({mean_t/total_batches*1000:.1f} ms/batch)", flush=True)

    # Liberar memoria GPU antes del siguiente combo
    del gen, net, opt_gen, opt_net, loader, dataset
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "dataset": dataset_name,
        "img_size": img_size,
        "channels": channels,
        "batch_size": cfg["batch_size"],
        "critic_iters": cfg["critic_iters"],
        "num_batches": total_batches,
        "measured_epochs": MEASURED_EPOCHS,
        "epoch_times": epoch_times,
        "mean_time_per_epoch": mean_t,
        "std_time_per_epoch": std_t,
        "ms_per_batch": mean_t / total_batches * 1000,
    }


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

def main():
    results = []
    t_start = time.perf_counter()

    for dataset_name, img_size, channels, data_dir in DATASETS:
        for model_name in MODEL_CONFIGS.keys():
            try:
                result = run_one_combo(model_name, dataset_name,
                                       img_size, channels, data_dir)
                results.append(result)
                # Guardar resultados parciales tras cada combo
                # (por si se interrumpe la ejecución)
                _save_results(results)
            except Exception as e:
                print(f"\n!!! ERROR en {model_name} / {dataset_name}: {e}",
                      flush=True)
                import traceback
                traceback.print_exc()

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Benchmarking completado en {elapsed_total/60:.1f} minutos")
    print(f"{'='*60}", flush=True)
    _save_results(results)


def _save_results(results):
    # JSON completo
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV resumen
    rows = []
    for r in results:
        rows.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "arch": MODEL_CONFIGS[r["model"]]["arch"],
            "loss": MODEL_CONFIGS[r["model"]]["loss"],
            "img_size": r["img_size"],
            "batch_size": r["batch_size"],
            "critic_iters": r["critic_iters"],
            "num_batches": r["num_batches"],
            "mean_time_per_epoch_s": r["mean_time_per_epoch"],
            "std_time_per_epoch_s": r["std_time_per_epoch"],
            "ms_per_batch": r["ms_per_batch"],
        })
    pd.DataFrame(rows).to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
