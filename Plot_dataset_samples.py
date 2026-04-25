import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# CONFIGURACIÓN
N_SAMPLES   = 10                   # imágenes por dataset
DATA_DIR    = "./data"             # MNIST, FashionMNIST y SVHN
OUT_PATH    = "muestra_datasets.png"

CELEBA_DIR  = "/home/pablo/Desktop/celeba-dataset./img_align_celeba"

BG      = "#F8F8F6"
C_LABEL = "#333333"

# Definición de datasets
DATASETS = [
    {
        "name":      "MNIST",
        "label":     "MNIST\n(escala de grises, 32×32)",
        "cls":       torchvision.datasets.MNIST,
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]),
        "grayscale": True,
        "kwargs":    {"train": True, "download": True},
        "root":      DATA_DIR,
        "skip":      False,
    },
    {
        "name":      "FashionMNIST",
        "label":     "FashionMNIST\n(escala de grises, 32×32)",
        "cls":       torchvision.datasets.FashionMNIST,
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]),
        "grayscale": True,
        "kwargs":    {"train": True, "download": True},
        "root":      DATA_DIR,
        "skip":      False,
    },
    {
        "name":      "SVHN",
        "label":     "SVHN\n(color, 32×32)",
        "cls":       torchvision.datasets.SVHN,
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]),
        "grayscale": False,
        "kwargs":    {"split": "train", "download": True},
        "root":      DATA_DIR,
        "skip":      False,
    },
    {
        "name":      "CelebA",
        "label":     "CelebA\n(color, 64×64)",
        "cls":       None,   # carga manual con PIL
        "transform": transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]),
        "grayscale": False,
        "kwargs":    {},
        "root":      "/home/pablo/Desktop/celeba-dataset./img_align_celeba/img_align_celeba",
        "skip":      False,
    },
]

# Carga de muestras
def get_samples(cfg, n):
    if cfg["cls"] is None:
        # Carga directa desde carpeta de imágenes
        from PIL import Image
        import os
        files = sorted([
            f for f in os.listdir(cfg["root"])
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        indices = torch.randperm(len(files))[:n]
        imgs = []
        for i in indices:
            img = Image.open(os.path.join(cfg["root"], files[i])).convert("RGB")
            imgs.append(cfg["transform"](img))
        return imgs
    else:
        dataset = cfg["cls"](root=cfg["root"], transform=cfg["transform"],
                             **cfg["kwargs"])
        indices = torch.randperm(len(dataset))[:n]
        return [dataset[i][0] for i in indices]

torch.manual_seed(42)
all_samples = []

for cfg in DATASETS:
    if cfg["skip"]:
        print(f"Omitiendo {cfg['name']} (CELEBA_DIR no configurado)")
        continue
    print(f"Cargando {cfg['name']}...", flush=True)
    samples = get_samples(cfg, N_SAMPLES)
    all_samples.append(cfg | {"samples": samples})

n_rows = len(all_samples)

# Figura
# CelebA ocupa el doble de alto (64px vs 32px)
n_rows = len(all_samples)

# Todas las filas tendrán el mismo peso visual.
fig_h = n_rows * 1.5 + 0.6  # Altura base proporcional al número de filas

fig = plt.figure(figsize=(N_SAMPLES * 1.15, fig_h), facecolor=BG)
gs  = gridspec.GridSpec(
    n_rows, N_SAMPLES,
    figure=fig,
    hspace=0.06, # Espacio vertical entre filas
    wspace=0.04, # Espacio horizontal entre columnas
    left=0.10, right=0.995,
    top=0.90,  bottom=0.02, # Bajamos un poco el 'top' para dar aire al título
)

# Dibujar imágenes
for row, ds in enumerate(all_samples):
    for col, img in enumerate(ds["samples"]):
        ax  = fig.add_subplot(gs[row, col])
        arr = img.permute(1, 2, 0).numpy()
        
        if ds["grayscale"]:
            ax.imshow(arr.squeeze(), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(arr, 0, 1))
            
        ax.set_xticks([]) # Quitamos ticks
        ax.set_yticks([])
        ax.axis("off")

# Forzar render para que get_position() devuelva valores correctos
fig.canvas.draw()

# Etiquetas de fila a la izquierda
for row, ds in enumerate(all_samples):
    axes_row = [fig.axes[row * N_SAMPLES + c] for c in range(N_SAMPLES)]
    bbox     = axes_row[0].get_position()
    y_c      = bbox.y0 + bbox.height / 2
    fig.text(
        0.075, y_c, ds["label"],
        ha="right", va="center",
        fontsize=8.5, color=C_LABEL, fontweight="bold",
        linespacing=1.4,
    )

fig.suptitle(
    "Muestras de los conjuntos de datos",
    fontsize=13, fontweight="bold", color=C_LABEL, y=0.97,
)

plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n✓ Guardado en: {OUT_PATH}")
