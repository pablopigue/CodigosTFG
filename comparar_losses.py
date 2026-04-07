"""
Gráficas comparativas de pérdidas separadas por familia.

Genera 3 gráficas por dataset:
  1. Modelos BCE: Loss_G y Loss_D de Vanilla, DCGAN, DCGAN+LS
     (corrige el factor x2 de Vanilla para igualar escala)
  2. Modelos Wasserstein: Loss_G y Loss_C de WGAN, WGAN-Conv, WGAN-GP
  3. WGAN-GP: curva de Gradient Penalty

USO:
  python comparar_losses.py --base_dir ./ResultadosCodigosGeneralizacion
  python comparar_losses.py --base_dir ./ResultadosCodigosGeneralizacion --datasets MNIST SVHN
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Modelos BCE: comparten la misma función de pérdida (BCELoss)
BCE_MODELOS = {
    "vanilla": {
        "nombre": "Vanilla GAN",
        "color": "#1f77b4",
        "linestyle": "--",
        "corregir_loss_d": True,  # loss_d está dividida entre 2
    },
    "dcgan": {
        "nombre": "DCGAN",
        "color": "#ff7f0e",
        "linestyle": "-",
        "corregir_loss_d": False,
    },
    "dcgan_ls": {
        "nombre": "DCGAN+LS",
        "color": "#d62728",
        "linestyle": "-.",
        "corregir_loss_d": False,
    },
}

# Modelos Wasserstein: comparten la misma función de pérdida
WASS_MODELOS = {
    "wgan": {
        "nombre": "WGAN",
        "color": "#2ca02c",
        "linestyle": "--",
    },
    "wgan_conv": {
        "nombre": "WGAN-Conv",
        "color": "#9467bd",
        "linestyle": "-",
    },
    "wgangp": {
        "nombre": "WGAN-GP",
        "color": "#e377c2",
        "linestyle": "-.",
    },
}

CARPETA_A_CLAVE = {
    "tfg_vanilla_gan": "vanilla",
    "tfg_dcgan":       "dcgan",
    "tfg_dcgan_ls":    "dcgan_ls",
    "tfg_wgan":        "wgan",
    "tfg_wgan_conv":   "wgan_conv",
    "tfg_wgangp":      "wgangp",
}

DATASETS_CONFIG = {
    "MNIST":        {"epochs": 80,  "runs": 10, "suffix": "mnist"},
    "FASHIONMNIST": {"epochs": 80,  "runs": 10, "suffix": "fashionmnist"},
    "SVHN":         {"epochs": 150, "runs": 10, "suffix": "svhn"},
    "CELEBA":       {"epochs": 40,  "runs": 5,  "suffix": "celeba"},
}


# ============================================================
# FUNCIONES
# ============================================================

def detectar_carpeta_modelo(dataset_dir, clave_modelo, dataset_suffix):
    for prefijo, clave in CARPETA_A_CLAVE.items():
        if clave == clave_modelo:
            ruta = os.path.join(dataset_dir, f"{prefijo}_{dataset_suffix}")
            if os.path.isdir(ruta):
                return ruta
    for d in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, d)
        if os.path.isdir(full) and clave_modelo.replace("_", "") in d.replace("_", ""):
            return full
    return None


def cargar_losses(modelo_dir):
    """Carga metrics_mean.csv y metrics_all_runs.csv."""
    mean_path = os.path.join(modelo_dir, "logs", "metrics_mean.csv")
    all_path = os.path.join(modelo_dir, "logs", "metrics_all_runs.csv")

    df_mean = pd.read_csv(mean_path) if os.path.exists(mean_path) else None
    df_all = pd.read_csv(all_path) if os.path.exists(all_path) else None

    return df_mean, df_all


def extraer_loss(df_mean, df_all, col, factor=1.0):
    """
    Extrae epoch, media y std de una columna de loss.
    factor: multiplicador para corregir escala (ej: 2.0 para Vanilla loss_d).
    """
    if df_mean is None or col not in df_mean.columns:
        return None

    epochs = df_mean['epoch'].values
    mean = df_mean[col].values * factor

    std = None
    if df_all is not None and col in df_all.columns:
        std_df = df_all.groupby('epoch')[col].std().reset_index()
        std_df.columns = ['epoch', 'std']
        # Alinear con las épocas del mean
        merged = pd.DataFrame({'epoch': epochs}).merge(std_df, on='epoch', how='left')
        std = merged['std'].values * factor

    return epochs, mean, std


def plot_losses_familia(datos, titulo, ylabel, filepath):
    """
    Genera una gráfica con 2 subplots: Loss G (arriba) y Loss D/C (abajo).

    datos: dict {clave: {nombre, color, linestyle,
                         g_epochs, g_mean, g_std,
                         d_epochs, d_mean, d_std}}
    """
    fig, (ax_g, ax_d) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for clave, d in datos.items():
        # Loss G
        if d.get('g_epochs') is not None:
            ax_g.plot(d['g_epochs'], d['g_mean'],
                      label=d['nombre'], color=d['color'],
                      linestyle=d['linestyle'], linewidth=2)
            if d.get('g_std') is not None:
                ax_g.fill_between(d['g_epochs'],
                                  d['g_mean'] - d['g_std'],
                                  d['g_mean'] + d['g_std'],
                                  color=d['color'], alpha=0.12)

        # Loss D/C
        if d.get('d_epochs') is not None:
            ax_d.plot(d['d_epochs'], d['d_mean'],
                      label=d['nombre'], color=d['color'],
                      linestyle=d['linestyle'], linewidth=2)
            if d.get('d_std') is not None:
                ax_d.fill_between(d['d_epochs'],
                                  d['d_mean'] - d['d_std'],
                                  d['d_mean'] + d['d_std'],
                                  color=d['color'], alpha=0.12)

    ax_g.set_ylabel('Pérdida del Generador', fontsize=11)
    ax_g.set_title(titulo, fontsize=14)
    ax_g.legend(fontsize=10)
    ax_g.grid(True, alpha=0.3)

    ax_d.set_xlabel('Épocas', fontsize=11)
    ax_d.set_ylabel(ylabel, fontsize=11)
    ax_d.legend(fontsize=10)
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {filepath}")


def plot_gradient_penalty(datos_gp, titulo, filepath):
    """Gráfica de Gradient Penalty de WGAN-GP."""
    if not datos_gp:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for clave, d in datos_gp.items():
        ax.plot(d['epochs'], d['mean'],
                label=d['nombre'], color=d['color'],
                linewidth=2)
        if d.get('std') is not None:
            ax.fill_between(d['epochs'],
                            d['mean'] - d['std'],
                            d['mean'] + d['std'],
                            color=d['color'], alpha=0.15)

    ax.set_xlabel('Épocas', fontsize=11)
    ax.set_ylabel('Gradient Penalty', fontsize=11)
    ax.set_title(titulo, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {filepath}")


# ============================================================
# PROCESAMIENTO POR DATASET
# ============================================================

def procesar_dataset(base_dir, dataset_name):
    cfg_ds = DATASETS_CONFIG[dataset_name]
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        print(f"\n[SKIP] No encontrado: {dataset_dir}")
        return

    print(f"\n{'='*60}")
    print(f" Procesando losses: {dataset_name}")
    print(f"{'='*60}")

    out_dir = os.path.join(dataset_dir, "_comparativas")
    os.makedirs(out_dir, exist_ok=True)

    # ---- MODELOS BCE ----
    datos_bce = {}
    for clave, cfg in BCE_MODELOS.items():
        modelo_dir = detectar_carpeta_modelo(dataset_dir, clave, cfg_ds["suffix"])
        if modelo_dir is None:
            print(f"  [SKIP] {clave} no encontrado")
            continue

        print(f"  Cargando: {cfg['nombre']}")
        df_mean, df_all = cargar_losses(modelo_dir)

        # Factor de corrección para Vanilla (loss_d está /2)
        factor_d = 2.0 if cfg.get("corregir_loss_d", False) else 1.0

        loss_g = extraer_loss(df_mean, df_all, 'loss_g')
        loss_d = extraer_loss(df_mean, df_all, 'loss_d', factor=factor_d)

        if loss_g is None:
            continue

        datos_bce[clave] = {
            'nombre': cfg['nombre'],
            'color': cfg['color'],
            'linestyle': cfg['linestyle'],
            'g_epochs': loss_g[0], 'g_mean': loss_g[1], 'g_std': loss_g[2],
            'd_epochs': loss_d[0] if loss_d else None,
            'd_mean': loss_d[1] if loss_d else None,
            'd_std': loss_d[2] if loss_d else None,
        }

        if cfg.get("corregir_loss_d"):
            print(f"    (loss_d de Vanilla corregida ×2 para igualar escala)")

    if datos_bce:
        plot_losses_familia(
            datos_bce,
            f"Pérdidas modelos BCE — {dataset_name} ({cfg_ds['runs']} runs)",
            "Pérdida del Discriminador",
            os.path.join(out_dir, "comparativa_losses_bce.png")
        )

    # ---- MODELOS WASSERSTEIN ----
    datos_wass = {}
    datos_gp = {}
    for clave, cfg in WASS_MODELOS.items():
        modelo_dir = detectar_carpeta_modelo(dataset_dir, clave, cfg_ds["suffix"])
        if modelo_dir is None:
            print(f"  [SKIP] {clave} no encontrado")
            continue

        print(f"  Cargando: {cfg['nombre']}")
        df_mean, df_all = cargar_losses(modelo_dir)

        loss_g = extraer_loss(df_mean, df_all, 'loss_g')
        loss_c = extraer_loss(df_mean, df_all, 'loss_c')

        if loss_g is None:
            continue

        datos_wass[clave] = {
            'nombre': cfg['nombre'],
            'color': cfg['color'],
            'linestyle': cfg['linestyle'],
            'g_epochs': loss_g[0], 'g_mean': loss_g[1], 'g_std': loss_g[2],
            'd_epochs': loss_c[0] if loss_c else None,
            'd_mean': loss_c[1] if loss_c else None,
            'd_std': loss_c[2] if loss_c else None,
        }

        # Gradient Penalty (solo WGAN-GP)
        if clave == "wgangp":
            gp = extraer_loss(df_mean, df_all, 'gp')
            if gp is not None:
                datos_gp[clave] = {
                    'nombre': cfg['nombre'],
                    'color': cfg['color'],
                    'epochs': gp[0], 'mean': gp[1], 'std': gp[2],
                }

    if datos_wass:
        plot_losses_familia(
            datos_wass,
            f"Pérdidas modelos Wasserstein — {dataset_name} ({cfg_ds['runs']} runs)",
            "Pérdida del Crítico",
            os.path.join(out_dir, "comparativa_losses_wasserstein.png")
        )

    if datos_gp:
        plot_gradient_penalty(
            datos_gp,
            f"Gradient Penalty (WGAN-GP) — {dataset_name} ({cfg_ds['runs']} runs)",
            os.path.join(out_dir, "comparativa_gradient_penalty.png")
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gráficas comparativas de pérdidas por familia (BCE / Wasserstein)."
    )
    parser.add_argument("--base_dir", type=str,
                        default="./ResultadosCodigosGeneralizacion")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    datasets = args.datasets or list(DATASETS_CONFIG.keys())

    for ds in datasets:
        if ds.upper() in DATASETS_CONFIG:
            procesar_dataset(base_dir, ds.upper())
        else:
            print(f"[WARN] Dataset '{ds}' no reconocido.")

    print(f"\n Completado.")


if __name__ == "__main__":
    main()
