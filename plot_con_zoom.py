"""
Genera la gráfica de pérdidas BCE con un recuadro (inset) de zoom
en el subplot del discriminador para ver Vanilla GAN y DCGAN+LS
que quedan aplastados por el colapso de DCGAN.

Requiere los CSV de metrics_all_runs de DCGAN y DCGAN+LS.
La Vanilla GAN se simula si no se dispone de su CSV (se puede añadir).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

MODELOS = {
    "vanilla": {
        "nombre": "Vanilla GAN",
        "color": "#1f77b4",
        "linestyle": "--",
        "carpeta": "tfg_vanilla_gan",
        "corregir_loss_d": True,  # loss_d está /2 en Vanilla
    },
    "dcgan": {
        "nombre": "DCGAN",
        "color": "#ff7f0e",
        "linestyle": "-",
        "carpeta": "tfg_dcgan",
        "corregir_loss_d": False,
    },
    "dcgan_ls": {
        "nombre": "DCGAN+LS",
        "color": "#d62728",
        "linestyle": "-.",
        "carpeta": "tfg_dcgan_ls",
        "corregir_loss_d": False,
    },
}

# ============================================================
# CONFIGURACIÓN DE DATASET
# Cambia BASE_DIR a la ruta de tu carpeta de resultados
# y DATASET al dataset que quieras graficar.
# ============================================================
BASE_DIR = "./ResultadosCodigosGeneralizacion"  # <-- CAMBIAR a tu ruta
DATASET = "MNIST"                                # <-- CAMBIAR si quieres otro dataset
DATASET_SUFFIX = DATASET.lower()                 # mnist, fashionmnist, svhn, celeba


# ============================================================
# CARGA Y PROCESADO
# ============================================================

def cargar_y_promediar(csv_path, factor_d=1.0):
    """Carga un CSV de all_runs y devuelve media±std por época."""
    df = pd.read_csv(csv_path)

    grouped = df.groupby('epoch').agg(
        loss_g_mean=('loss_g', 'mean'),
        loss_g_std=('loss_g', 'std'),
        loss_d_mean=('loss_d', 'mean'),
        loss_d_std=('loss_d', 'std'),
    ).reset_index()

    grouped['loss_d_mean'] *= factor_d
    grouped['loss_d_std'] *= factor_d

    return grouped


# ============================================================
# GRÁFICA
# ============================================================

def plot_bce_con_inset(datos, filepath):
    fig, (ax_g, ax_g_zoom, ax_d, ax_zoom) = plt.subplots(4, 1, figsize=(12, 14),
                                                           height_ratios=[1, 0.8, 1, 0.8])

    # --- Subplot 1: Loss G (escala completa) ---
    for clave, d in datos.items():
        ax_g.plot(d['epochs'], d['g_mean'],
                  label=d['nombre'], color=d['color'],
                  linestyle=d['linestyle'], linewidth=2)
        if d['g_std'] is not None:
            ax_g.fill_between(d['epochs'],
                              d['g_mean'] - d['g_std'],
                              d['g_mean'] + d['g_std'],
                              color=d['color'], alpha=0.12)

    ax_g.set_ylabel('Pérdida del Generador', fontsize=12)
    ax_g.set_title('Pérdidas modelos BCE — MNIST (10 runs)', fontsize=14)
    ax_g.legend(fontsize=10, loc='upper left')
    ax_g.grid(True, alpha=0.3)
    ax_g.set_xlim(0, 80)

    # --- Subplot 2: Zoom Loss G ---
    for clave, d in datos.items():
        ax_g_zoom.plot(d['epochs'], d['g_mean'],
                       label=d['nombre'], color=d['color'],
                       linestyle=d['linestyle'], linewidth=2)
        if d['g_std'] is not None:
            ax_g_zoom.fill_between(d['epochs'],
                                   d['g_mean'] - d['g_std'],
                                   d['g_mean'] + d['g_std'],
                                   color=d['color'], alpha=0.12)

    ax_g_zoom.set_ylim(-1, 10)
    ax_g_zoom.set_xlim(0, 80)
    ax_g_zoom.set_ylabel('Pérdida del Generador\n(zoom)', fontsize=12)
    ax_g_zoom.set_title('Zoom pérdida del generador (rango -1 a 10)', fontsize=11, fontstyle='italic')
    ax_g_zoom.legend(fontsize=10, loc='upper left')
    ax_g_zoom.grid(True, alpha=0.3)

    # --- Subplot 3: Loss D (escala completa con colapso) ---
    for clave, d in datos.items():
        ax_d.plot(d['epochs'], d['d_mean'],
                  label=d['nombre'], color=d['color'],
                  linestyle=d['linestyle'], linewidth=2)
        if d['d_std'] is not None:
            ax_d.fill_between(d['epochs'],
                              d['d_mean'] - d['d_std'],
                              d['d_mean'] + d['d_std'],
                              color=d['color'], alpha=0.12)

    ax_d.set_ylabel('Pérdida del Discriminador', fontsize=12)
    ax_d.legend(fontsize=10, loc='upper left')
    ax_d.grid(True, alpha=0.3)
    ax_d.set_xlim(0, 80)

    # --- Subplot 4: Zoom del discriminador ---
    for clave, d in datos.items():
        ax_zoom.plot(d['epochs'], d['d_mean'],
                     label=d['nombre'], color=d['color'],
                     linestyle=d['linestyle'], linewidth=2)
        if d['d_std'] is not None:
            ax_zoom.fill_between(d['epochs'],
                                 d['d_mean'] - d['d_std'],
                                 d['d_mean'] + d['d_std'],
                                 color=d['color'], alpha=0.12)

    ax_zoom.set_ylim(-0.2, 2.5)
    ax_zoom.set_xlim(0, 80)
    ax_zoom.set_xlabel('Épocas', fontsize=12)
    ax_zoom.set_ylabel('Pérdida del Discriminador\n(zoom)', fontsize=12)
    ax_zoom.set_title('Zoom pérdida del discriminador (rango 0–2,5)', fontsize=11, fontstyle='italic')
    ax_zoom.legend(fontsize=10, loc='upper left')
    ax_zoom.grid(True, alpha=0.3)

    plt.subplots_adjust(hspace=0.35)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    datos = {}
    dataset_dir = os.path.join(BASE_DIR, DATASET)

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] No encontrado: {dataset_dir}")
        print(f"  Ajusta BASE_DIR y DATASET al principio del script.")
        return

    for clave, cfg in MODELOS.items():
        # Construir ruta: BASE_DIR/DATASET/tfg_xxx_dataset/logs/metrics_all_runs.csv
        modelo_dir = os.path.join(dataset_dir, f"{cfg['carpeta']}_{DATASET_SUFFIX}")
        csv_path = os.path.join(modelo_dir, "logs", "metrics_all_runs.csv")

        if not os.path.exists(csv_path):
            print(f"[SKIP] {cfg['nombre']}: no encontrado en {csv_path}")
            continue

        factor_d = 2.0 if cfg['corregir_loss_d'] else 1.0
        grouped = cargar_y_promediar(csv_path, factor_d)

        datos[clave] = {
            'nombre': cfg['nombre'],
            'color': cfg['color'],
            'linestyle': cfg['linestyle'],
            'epochs': grouped['epoch'].values,
            'g_mean': grouped['loss_g_mean'].values,
            'g_std': grouped['loss_g_std'].values,
            'd_mean': grouped['loss_d_mean'].values,
            'd_std': grouped['loss_d_std'].values,
        }

        print(f"Cargado: {cfg['nombre']} ({len(grouped)} épocas) desde {modelo_dir}")

    if datos:
        out_dir = os.path.join(dataset_dir, "_comparativas")
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, "imagen_loss_nueva.png")
        plot_bce_con_inset(datos, filepath)
    else:
        print("[ERROR] No se cargó ningún modelo.")


if __name__ == "__main__":
    main()