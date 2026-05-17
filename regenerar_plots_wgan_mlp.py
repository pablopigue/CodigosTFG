"""
Regenera los plots de WGAN (renombrado a WGAN-MLP) a partir de los CSV
`metrics_all_runs.csv` ubicados dentro de las carpetas de resultados.

Estructura esperada (relativa a BASE_DIR):
    MNIST/tfg_wgan_mnist/logs/metrics_all_runs.csv
    FASHIONMNIST/tfg_wgan_fashionmnist/logs/metrics_all_runs.csv
    SVHN/tfg_wgan_svhn/logs/metrics_all_runs.csv
    CELEBA/tfg_wgan_celeba/logs/metrics_all_runs.csv

USO:
    python regenerar_plots_wgan_mlp.py
    # Los PNG salen en ./nuevos_wganmlp/<dataset>/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_NAME = "WGAN-MLP"

# Ruta raíz donde están MNIST/, FASHIONMNIST/, SVHN/, CELEBA/
BASE_DIR = "/home/pablo/Desktop/carpetalocal/CodigosTFG/ResultadosCodigosGeneralizacion"

# (nombre que aparece en los títulos, subcarpeta del dataset, subcarpeta del modelo)
DATASETS = [
    ("MNIST",         "MNIST",        "tfg_wgan_mnist"),
    ("Fashion-MNIST", "FASHIONMNIST", "tfg_wgan_fashionmnist"),
    ("SVHN",          "SVHN",         "tfg_wgan_svhn"),
    ("CelebA",        "CELEBA",       "tfg_wgan_celeba"),
]

# Carpeta de salida (con subcarpetas por dataset)
OUT_ROOT = "./nuevos_wganmlp"

plt.switch_backend('agg')


# ============================================================
# REPLICA DE utils.save_plot
# ============================================================

def save_plot(x, ys, labels, colors, title, xlabel, ylabel, filepath,
              stds=None, markers=None):
    plt.figure(figsize=(10, 5))
    for i, (y, label, color) in enumerate(zip(ys, labels, colors)):
        marker = markers[i] if markers else None
        plt.plot(x, y, label=label, color=color, marker=marker)
        if stds is not None and stds[i] is not None:
            plt.fill_between(x, y - stds[i], y + stds[i],
                             color=color, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()


# ============================================================
# PROCESAMIENTO POR DATASET
# ============================================================

def procesar_dataset(dataset_name, dataset_folder, model_folder):
    csv_path = os.path.join(BASE_DIR, dataset_folder, model_folder,
                            "logs", "metrics_all_runs.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] No encontrado: {csv_path}")
        return

    out_dir = os.path.join(OUT_ROOT,
                           dataset_name.lower().replace("-", "").replace(" ", ""))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {dataset_name} ===")
    print(f"  CSV: {csv_path}")
    df_all = pd.read_csv(csv_path)

    loss_disc_col   = 'loss_c'
    loss_disc_label = 'Pérdida del Crítico'
    loss_ylabel     = 'Pérdida Wasserstein'

    # AGREGADOS (media y std por época)
    loss_cols = ['loss_g', loss_disc_col]
    df_mean_loss = df_all.groupby('epoch')[loss_cols].mean().reset_index()
    df_std_loss  = df_all.groupby('epoch')[loss_cols].std().reset_index()

    df_metrics = df_all.dropna(subset=['fid'])
    df_mean_metrics = (df_metrics.groupby('epoch')[['fid', 'is_mean']]
                       .mean().reset_index())
    df_std_metrics  = (df_metrics.groupby('epoch')[['fid', 'is_mean']]
                       .std().reset_index())

    save_plot(
        x=df_mean_loss['epoch'],
        ys=[df_mean_loss['loss_g'], df_mean_loss[loss_disc_col]],
        labels=['Pérdida del Generador', loss_disc_label],
        colors=['blue', 'orange'],
        title=f'Curvas de Aprendizaje Promediadas {MODEL_NAME} - {dataset_name}',
        xlabel='Épocas', ylabel=loss_ylabel,
        filepath=os.path.join(out_dir, 'training_losses_mean.png'),
        stds=[df_std_loss['loss_g'], df_std_loss[loss_disc_col]]
    )

    save_plot(
        x=df_mean_metrics['epoch'], ys=[df_mean_metrics['fid']],
        labels=['Puntuación FID'], colors=['green'],
        title=f'Evolución de la Calidad FID Promediada {MODEL_NAME} - {dataset_name}',
        xlabel='Épocas', ylabel='FID',
        filepath=os.path.join(out_dir, 'fid_metric_mean.png'),
        stds=[df_std_metrics['fid']], markers=['o']
    )

    save_plot(
        x=df_mean_metrics['epoch'], ys=[df_mean_metrics['is_mean']],
        labels=['Puntuación Inception (media)'], colors=['purple'],
        title=f'Evolución del Inception Score Promediado {MODEL_NAME} - {dataset_name}',
        xlabel='Épocas', ylabel='IS',
        filepath=os.path.join(out_dir, 'is_metric_mean.png'),
        stds=[df_std_metrics['is_mean']], markers=['o']
    )

    # RUN 1
    df_run = df_all[df_all['run'] == 1].copy()
    df_run_metrics = df_run.dropna(subset=['fid'])

    save_plot(
        x=df_run['epoch'], ys=[df_run['loss_g'], df_run[loss_disc_col]],
        labels=['Pérdida del Generador', loss_disc_label],
        colors=['blue', 'orange'],
        title=f'Curvas de Aprendizaje {MODEL_NAME} - {dataset_name}',
        xlabel='Épocas', ylabel=loss_ylabel,
        filepath=os.path.join(out_dir, 'training_losses_run1.png')
    )
    save_plot(
        x=df_run_metrics['epoch'], ys=[df_run_metrics['fid']],
        labels=['Puntuación FID'], colors=['green'],
        title=f'Evolución de la Calidad FID - {dataset_name}',
        xlabel='Épocas', ylabel='FID',
        filepath=os.path.join(out_dir, 'fid_metric_run1.png'),
        markers=['o']
    )
    save_plot(
        x=df_run_metrics['epoch'], ys=[df_run_metrics['is_mean']],
        labels=['Puntuación Inception (media)'], colors=['purple'],
        title=f'Evolución del Inception Score - {dataset_name}',
        xlabel='Épocas', ylabel='IS',
        filepath=os.path.join(out_dir, 'is_metric_run1.png'),
        stds=[df_run_metrics['is_std']], markers=['o']
    )

    print(f"  -> 6 PNG guardados en {out_dir}/")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    for nombre, dataset_folder, model_folder in DATASETS:
        procesar_dataset(nombre, dataset_folder, model_folder)
    print(f"\nCompletado. Plots en {OUT_ROOT}/")