import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN GENERAL Y DEL DATASET
plt.switch_backend('agg')

# Opciones válidas: "MNIST", "FashionMNIST", "SVHN"
DATASET_NAME = "MNIST"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0002
BATCH_SIZE = 128
Z_DIM = 100
SAVE_IMG_FREQ = 5
CALC_METRICS_FREQ = 5
NUM_RUNS = 10
NUM_EVAL_IMAGES = 10000
IMG_SIZE = 32   # Unificado a 32 en todos los modelos para que el FID comparable
if DATASET_NAME in ["MNIST", "FashionMNIST"]:
    CHANNELS = 1
    EPOCHS = 80
elif DATASET_NAME == "SVHN":
    CHANNELS = 3
    EPOCHS = 150
else:
    raise ValueError(f"Dataset '{DATASET_NAME}' no reconocido. Opciones válidas: MNIST, FashionMNIST, SVHN.")

IMG_DIM = CHANNELS * IMG_SIZE * IMG_SIZE

EXPERIMENT_DIR = f"/mnt/homeGPU/pablomarpa/CodigosTFG/tfg_vanilla_gan_{DATASET_NAME.lower()}"

os.makedirs(f"{EXPERIMENT_DIR}/images", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/plots", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/logs", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/models", exist_ok=True)

print(f"Iniciando entrenamiento en: {DEVICE} con dataset {DATASET_NAME}. Ejecuciones totales: {NUM_RUNS}", flush=True)

# ==========================================
# 2. CARGA DE DATOS
# ==========================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*CHANNELS, [0.5]*CHANNELS)
])

if DATASET_NAME == "MNIST":
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
elif DATASET_NAME == "FashionMNIST":
    dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
elif DATASET_NAME == "SVHN":
    dataset = torchvision.datasets.SVHN(root="./data", split='train', download=True, transform=transform)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ==========================================
# 3. SUBCONJUNTO FIJO DE EVALUACIÓN PARA FID
# Seed fija SOLO para el muestreo del subconjunto de evaluación.
# Garantiza representatividad de todas las clases (evita sesgo por orden
# en datasets como FashionMNIST) y que el subconjunto sea idéntico entre
# runs y entre modelos distintos.
# Se resetea inmediatamente después para que los runs sean independientes.
# ==========================================
torch.manual_seed(42)
indices = torch.randperm(len(dataset))[:min(NUM_EVAL_IMAGES, len(dataset))]
eval_subset = Subset(dataset, indices)
eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

torch.seed()
np.random.seed()

print("Precargando subconjunto fijo de imágenes reales para evaluación FID/IS...", flush=True)

real_eval_images = []
with torch.no_grad():
    for data in eval_loader:
        real_imgs = data[0].to(DEVICE)
        if CHANNELS == 1:
            real_uint8 = ((real_imgs * 0.5 + 0.5) * 255).type(torch.uint8).repeat(1, 3, 1, 1)
        else:
            real_uint8 = ((real_imgs * 0.5 + 0.5) * 255).type(torch.uint8)
        real_eval_images.append(real_uint8)

print(f"Subconjunto fijo listo: {len(real_eval_images) * BATCH_SIZE} imágenes reales ({len(real_eval_images)} batches).", flush=True)

# ==========================================
# 4. DEFINICIÓN DE CLASES
# ==========================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(IMG_DIM, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
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
# 5. FUNCIÓN AUXILIAR PARA GUARDAR GRÁFICAS
# ==========================================
def save_plot(x, ys, labels, colors, title, xlabel, ylabel, filepath, stds=None, markers=None):
    plt.figure(figsize=(10, 5))
    for i, (y, label, color) in enumerate(zip(ys, labels, colors)):
        marker = markers[i] if markers else None
        plt.plot(x, y, label=label, color=color, marker=marker)
        if stds is not None and stds[i] is not None:
            plt.fill_between(x, y - stds[i], y + stds[i], color=color, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

# ==========================================
# 6. BUCLE MULTI-RUN
# ==========================================
all_runs_data = []

for run in range(NUM_RUNS):
    print(f"\n{'='*40}")
    print(f" INICIANDO EJECUCIÓN {run + 1} DE {NUM_RUNS}")
    print(f"{'='*40}\n", flush=True)

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    # betas=(0.5, 0.999): estándar para GANs. Referencia: Radford et al. (2015), arXiv:1511.06434
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
    is_metric = InceptionScore().to(DEVICE)
    fixed_noise = torch.randn(32, Z_DIM).to(DEVICE)

    history = {"epoch": [], "loss_g": [], "loss_d": [], "fid": [], "is_mean": [], "is_std": []}

    for epoch in range(EPOCHS):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        gen.train()
        disc.train()
        for batch_idx, data in enumerate(loader):
            real = data[0].view(-1, IMG_DIM).to(DEVICE)
            batch_size_curr = real.shape[0]

            disc_real = disc(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            noise = torch.randn(batch_size_curr, Z_DIM).to(DEVICE)
            fake = gen(noise)
            disc_fake = disc(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            output = disc(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            epoch_loss_g += loss_gen.item()
            epoch_loss_d += loss_disc.item()

        avg_loss_g = epoch_loss_g / len(loader)
        avg_loss_d = epoch_loss_d / len(loader)

        current_fid = np.nan
        current_is_mean = np.nan
        current_is_std = np.nan

        if (epoch + 1) % CALC_METRICS_FREQ == 0 or epoch == 0 or epoch == EPOCHS - 1:
            gen.eval()
            fid_metric.reset()
            is_metric.reset()

            with torch.no_grad():
                for real_uint8 in real_eval_images:
                    fid_metric.update(real_uint8, real=True)

                for real_uint8 in real_eval_images:
                    batch_size_curr = real_uint8.shape[0]
                    noise = torch.randn(batch_size_curr, Z_DIM).to(DEVICE)
                    fake_eval = gen(noise).view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)
                    if CHANNELS == 1:
                        fake_uint8 = ((fake_eval * 0.5 + 0.5) * 255).type(torch.uint8).repeat(1, 3, 1, 1)
                    else:
                        fake_uint8 = ((fake_eval * 0.5 + 0.5) * 255).type(torch.uint8)
                    fid_metric.update(fake_uint8, real=False)
                    is_metric.update(fake_uint8)

            current_fid = fid_metric.compute().item()
            is_mean, is_std = is_metric.compute()
            current_is_mean = is_mean.item()
            current_is_std = is_std.item()

        if not np.isnan(current_fid):
            print(f"Ejecución [{run+1}/{NUM_RUNS}] - Época [{epoch+1}/{EPOCHS}] "
                  f"Pérdida D: {avg_loss_d:.4f} | Pérdida G: {avg_loss_g:.4f} | "
                  f"FID: {current_fid:.2f} | IS: {current_is_mean:.2f} ± {current_is_std:.2f}", flush=True)
        else:
            print(f"Ejecución [{run+1}/{NUM_RUNS}] - Época [{epoch+1}/{EPOCHS}] "
                  f"Pérdida D: {avg_loss_d:.4f} | Pérdida G: {avg_loss_g:.4f} | "
                  f"FID: --- | IS: ---", flush=True)

        history["epoch"].append(epoch + 1)
        history["loss_g"].append(avg_loss_g)
        history["loss_d"].append(avg_loss_d)
        history["fid"].append(current_fid)
        history["is_mean"].append(current_is_mean)
        history["is_std"].append(current_is_std)

        if run == 0 and ((epoch + 1) % SAVE_IMG_FREQ == 0 or epoch == 0):
            with torch.no_grad():
                fake_display = gen(fixed_noise).reshape(-1, CHANNELS, IMG_SIZE, IMG_SIZE)
                fake_display = fake_display * 0.5 + 0.5
                grid = torchvision.utils.make_grid(fake_display, nrow=8, normalize=False)
                plt.figure(figsize=(8, 4))
                if CHANNELS == 1:
                    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
                else:
                    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.title(f"Imágenes Generadas ({DATASET_NAME}) - Época {epoch+1}")
                plt.savefig(f"{EXPERIMENT_DIR}/images/epoch_{epoch+1}.png")
                plt.close()

    df_run = pd.DataFrame(history)
    df_run['run'] = run + 1
    all_runs_data.append(df_run)

    if run == 0:
        torch.save(gen.state_dict(), f"{EXPERIMENT_DIR}/models/generator_final_run1.pth")
        torch.save(disc.state_dict(), f"{EXPERIMENT_DIR}/models/discriminator_final_run1.pth")
        df_run.to_csv(f"{EXPERIMENT_DIR}/logs/metrics_run1.csv", index=False)

        df_run_metrics = df_run.dropna(subset=['fid'])

        save_plot(
            x=df_run['epoch'], ys=[df_run['loss_g'], df_run['loss_d']],
            labels=['Pérdida del Generador', 'Pérdida del Discriminador'],
            colors=['blue', 'orange'],
            title=f'Curvas de Aprendizaje Vanilla GAN - {DATASET_NAME}',
            xlabel='Épocas', ylabel='Pérdida',
            filepath=f"{EXPERIMENT_DIR}/plots/training_losses_run1.png"
        )
        save_plot(
            x=df_run_metrics['epoch'], ys=[df_run_metrics['fid']],
            labels=['Puntuación FID'], colors=['green'],
            title=f'Evolución de la Calidad FID - {DATASET_NAME}',
            xlabel='Épocas', ylabel='FID',
            filepath=f"{EXPERIMENT_DIR}/plots/fid_metric_run1.png", markers=['o']
        )
        save_plot(
            x=df_run_metrics['epoch'], ys=[df_run_metrics['is_mean']],
            labels=['Puntuación Inception (media)'], colors=['purple'],
            title=f'Evolución del Inception Score - {DATASET_NAME}',
            xlabel='Épocas', ylabel='IS',
            filepath=f"{EXPERIMENT_DIR}/plots/is_metric_run1.png",
            stds=[df_run_metrics['is_std']], markers=['o']
        )

# ==========================================
# 7. POST-PROCESADO Y PROMEDIADO FINAL
# ==========================================
print("\nGenerando gráficas promediadas...", flush=True)

df_all = pd.concat(all_runs_data, ignore_index=True)
df_all.to_csv(f"{EXPERIMENT_DIR}/logs/metrics_all_runs.csv", index=False)

df_mean_loss = df_all.groupby('epoch')[['loss_g', 'loss_d']].mean().reset_index()
df_std_loss = df_all.groupby('epoch')[['loss_g', 'loss_d']].std().reset_index()

df_metrics = df_all.dropna(subset=['fid'])
df_mean_metrics = df_metrics.groupby('epoch')[['fid', 'is_mean']].mean().reset_index()
df_std_metrics = df_metrics.groupby('epoch')[['fid', 'is_mean']].std().reset_index()

df_mean_final = pd.merge(df_mean_loss, df_mean_metrics, on='epoch', how='left')
df_mean_final.to_csv(f"{EXPERIMENT_DIR}/logs/metrics_mean.csv", index=False)

save_plot(
    x=df_mean_loss['epoch'], ys=[df_mean_loss['loss_g'], df_mean_loss['loss_d']],
    labels=['Pérdida del Generador', 'Pérdida del Discriminador'],
    colors=['blue', 'orange'],
    title=f'Curvas de Aprendizaje Promediadas - {DATASET_NAME}',
    xlabel='Épocas', ylabel='Pérdida',
    filepath=f"{EXPERIMENT_DIR}/plots/training_losses_mean.png",
    stds=[df_std_loss['loss_g'], df_std_loss['loss_d']]
)
save_plot(
    x=df_mean_metrics['epoch'], ys=[df_mean_metrics['fid']],
    labels=['Puntuación FID'], colors=['green'],
    title=f'Evolución de la Calidad FID Promediada - {DATASET_NAME}',
    xlabel='Épocas', ylabel='FID',
    filepath=f"{EXPERIMENT_DIR}/plots/fid_metric_mean.png",
    stds=[df_std_metrics['fid']], markers=['o']
)
save_plot(
    x=df_mean_metrics['epoch'], ys=[df_mean_metrics['is_mean']],
    labels=['Puntuación Inception (media)'], colors=['purple'],
    title=f'Evolución del Inception Score Promediado - {DATASET_NAME}',
    xlabel='Épocas', ylabel='IS',
    filepath=f"{EXPERIMENT_DIR}/plots/is_metric_mean.png",
    stds=[df_std_metrics['is_mean']], markers=['o']
)

print(f"Entrenamiento de {NUM_RUNS} ejecuciones finalizado. Todo guardado en: {EXPERIMENT_DIR}", flush=True)