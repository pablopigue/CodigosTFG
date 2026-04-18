"""
Utilidades compartidas para los experimentos de GANs del TFG.

Este módulo centraliza la lógica común a todos los modelos (Vanilla GAN,
DCGAN, DCGAN+LS, WGAN, WGAN-Conv, WGAN-GP) para evitar duplicación de
código entre los scripts de entrenamiento.

Bloques:
    A. Carga de datos y transformaciones.
    B. Subconjunto fijo de evaluación FID/IS.
    C. Cálculo de métricas FID/IS.
    D. Inicialización de pesos y guardado de gráficas/imágenes.
    E. Post-procesado multi-run y creación de directorios.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
import matplotlib.pyplot as plt


# ============================================================
# BLOQUE A: CARGA DE DATOS
# ============================================================

def build_transform(img_size, channels):
    """Transformación estándar: resize + tensor + normalización a [-1, 1]."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * channels, [0.5] * channels)
    ])


def load_dataset(dataset_name, img_size, channels, data_dir=None):
    """
    Carga un dataset por nombre.

    - Para MNIST/FashionMNIST/SVHN: usa torchvision con descarga automática.
    - Para CelebA: usa ImageFolder sobre `data_dir`.

    Parameters
    ----------
    dataset_name : str
        Uno de {"MNIST", "FashionMNIST", "SVHN", "CelebA"}.
    img_size : int
        Resolución objetivo (se aplica Resize).
    channels : int
        Número de canales (1 o 3).
    data_dir : str, optional
        Directorio raíz para CelebA (requerido solo en ese caso).

    Returns
    -------
    torch.utils.data.Dataset
    """
    transform = build_transform(img_size, channels)

    if dataset_name == "MNIST":
        return torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name == "FashionMNIST":
        return torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name == "SVHN":
        return torchvision.datasets.SVHN(
            root="./data", split='train', download=True, transform=transform
        )
    elif dataset_name == "CelebA":
        if data_dir is None:
            raise ValueError("CelebA requiere un data_dir válido.")
        return torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' no reconocido. "
            f"Opciones válidas: MNIST, FashionMNIST, SVHN, CelebA."
        )


# ============================================================
# BLOQUE B: SUBCONJUNTO FIJO DE EVALUACIÓN FID/IS
# ============================================================

def build_fixed_eval_set(dataset, batch_size, num_eval_images, device,
                         channels, seed=42):
    """
    Construye el subconjunto fijo para evaluación FID/IS.

    Se fija la semilla SOLO para el muestreo del subconjunto, garantizando
    que todos los modelos y runs se evalúan contra las mismas imágenes
    reales. Inmediatamente después se resetea la semilla para que los
    runs de entrenamiento sean independientes.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    batch_size : int
    num_eval_images : int
        Número de imágenes a incluir en el subconjunto.
    device : torch.device
    channels : int
        1 o 3. Si es 1, las imágenes se replican a 3 canales para
        ser compatibles con InceptionV3.
    seed : int, default=42
        Semilla para el muestreo del subconjunto.

    Returns
    -------
    list of torch.Tensor
        Lista de batches de imágenes reales en formato uint8, listas para
        ser pasadas a `fid_metric.update(..., real=True)`.
    """
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:min(num_eval_images, len(dataset))]
    eval_subset = Subset(dataset, indices)
    eval_loader = DataLoader(
        eval_subset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Resetear la semilla para que los runs de entrenamiento sean independientes.
    torch.seed()
    np.random.seed(None)

    print("Precargando subconjunto fijo de imágenes reales para evaluación "
          "FID/IS...", flush=True)

    real_eval_images = []
    with torch.no_grad():
        for data in eval_loader:
            real_imgs = data[0].to(device)
            real_uint8 = ((real_imgs * 0.5 + 0.5) * 255).type(torch.uint8)
            if channels == 1:
                real_uint8 = real_uint8.repeat(1, 3, 1, 1)
            real_eval_images.append(real_uint8)

    print(f"Subconjunto fijo listo: {len(real_eval_images) * batch_size} "
          f"imágenes reales ({len(real_eval_images)} batches).", flush=True)

    return real_eval_images


# ============================================================
# BLOQUE C: CÁLCULO DE MÉTRICAS FID E IS
# ============================================================

def compute_fid_is(gen, real_eval_images, z_dim, device, channels, img_size,
                   flatten_output=False):
    """
    Calcula FID e IS sobre el subconjunto fijo de imágenes reales y un
    conjunto de imágenes generadas del mismo tamaño.

    Parameters
    ----------
    gen : nn.Module
        Generador. DEBE estar en modo eval antes de llamar a esta función.
    real_eval_images : list of torch.Tensor
        Batches precalculados de imágenes reales en uint8 (producidos por
        `build_fixed_eval_set`).
    z_dim : int
        Dimensión del espacio latente.
    device : torch.device
    channels : int
        1 o 3.
    img_size : int
        Resolución de las imágenes (necesario solo si flatten_output=True).
    flatten_output : bool, default=False
        True si el generador devuelve un tensor plano (MLP);
        False si devuelve un tensor 4D (CNN).

    Returns
    -------
    (fid, is_mean, is_std) : tuple of float
    """
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)

    with torch.no_grad():
        # Actualizar FID con las imágenes reales
        for real_uint8 in real_eval_images:
            fid_metric.update(real_uint8, real=True)

        # Generar y actualizar FID/IS con imágenes fake
        for real_uint8 in real_eval_images:
            batch_size_curr = real_uint8.shape[0]
            noise = torch.randn(batch_size_curr, z_dim).to(device)
            fake = gen(noise)

            if flatten_output:
                fake = fake.view(-1, channels, img_size, img_size)

            fake_uint8 = ((fake * 0.5 + 0.5) * 255).type(torch.uint8)
            if channels == 1:
                fake_uint8 = fake_uint8.repeat(1, 3, 1, 1)

            fid_metric.update(fake_uint8, real=False)
            is_metric.update(fake_uint8)

    fid = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    return fid, is_mean.item(), is_std.item()


# ============================================================
# BLOQUE D: INICIALIZACIÓN Y GUARDADO DE GRÁFICAS/IMÁGENES
# ============================================================

def weights_init_dcgan(m):
    """
    Inicialización estándar de DCGAN.

    - Capas convolucionales: N(0, 0.02).
    - Capas de normalización (BatchNorm/GroupNorm): gamma ~ N(1, 0.02), beta = 0.

    Referencia: Radford et al. (2015), arXiv:1511.06434.
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname or 'GroupNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_plot(x, ys, labels, colors, title, xlabel, ylabel, filepath,
              stds=None, markers=None):
    """
    Guarda una gráfica multi-serie con bandas opcionales de desviación estándar.

    Parameters
    ----------
    x : array-like
        Eje horizontal común a todas las series.
    ys : list of array-like
        Series a graficar.
    labels : list of str
        Etiquetas de leyenda para cada serie.
    colors : list of str
        Colores para cada serie.
    title, xlabel, ylabel : str
    filepath : str
        Ruta donde guardar el PNG.
    stds : list of array-like or None
        Si se proporciona, dibuja bandas y +/- std alrededor de cada serie.
    markers : list of str or None
        Marcadores para cada serie (ej: 'o', 's').
    """
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


def save_sample_images(gen, fixed_noise, channels, img_size, filepath, title,
                       flatten_output=False):
    """
    Guarda una cuadrícula de imágenes generadas a partir de un ruido fijo.

    Parameters
    ----------
    gen : nn.Module
        Generador.
    fixed_noise : torch.Tensor
        Tensor de ruido fijo de dimensión (N, z_dim) usado para visualizar
        la evolución del generador.
    channels : int
    img_size : int
    filepath : str
    title : str
    flatten_output : bool, default=False
        True si el generador devuelve un tensor plano (MLP).
    """
    with torch.no_grad():
        fake = gen(fixed_noise)
        if flatten_output:
            fake = fake.reshape(-1, channels, img_size, img_size)
        fake = fake * 0.5 + 0.5
        grid = torchvision.utils.make_grid(fake, nrow=8, normalize=False)

        plt.figure(figsize=(8, 8))
        if channels == 1:
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        else:
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(title)
        plt.savefig(filepath)
        plt.close()


# ============================================================
# BLOQUE E: POST-PROCESADO Y DIRECTORIOS
# ============================================================

def make_experiment_dirs(experiment_dir):
    """Crea los subdirectorios estándar del experimento."""
    for sub in ['images', 'plots', 'logs', 'models']:
        os.makedirs(f"{experiment_dir}/{sub}", exist_ok=True)


def save_run_artifacts(gen, disc_or_critic, df_run, experiment_dir,
                       model_name, dataset_name, critic_name='discriminator'):
    """
    Guarda los artefactos de la primera ejecución: pesos, CSV y gráficas
    individuales del run.

    Parameters
    ----------
    gen, disc_or_critic : nn.Module
    df_run : pd.DataFrame
    experiment_dir : str
    model_name : str
        Nombre para los títulos de las gráficas (ej: "DCGAN", "WGAN-GP").
    dataset_name : str
    critic_name : str, default='discriminator'
        'discriminator' o 'critic'. Se usa en el nombre del fichero de pesos.
    """
    torch.save(gen.state_dict(),
               f"{experiment_dir}/models/generator_final_run1.pth")
    torch.save(disc_or_critic.state_dict(),
               f"{experiment_dir}/models/{critic_name}_final_run1.pth")
    df_run.to_csv(f"{experiment_dir}/logs/metrics_run1.csv", index=False)

    df_metrics = df_run.dropna(subset=['fid'])

    # Determinar la columna de pérdida del discriminador/crítico
    loss_disc_col = 'loss_c' if 'loss_c' in df_run.columns else 'loss_d'
    loss_disc_label = ('Pérdida del Crítico'
                       if loss_disc_col == 'loss_c'
                       else 'Pérdida del Discriminador')
    loss_ylabel = ('Pérdida Wasserstein'
                   if loss_disc_col == 'loss_c' else 'Pérdida')

    save_plot(
        x=df_run['epoch'], ys=[df_run['loss_g'], df_run[loss_disc_col]],
        labels=['Pérdida del Generador', loss_disc_label],
        colors=['blue', 'orange'],
        title=f'Curvas de Aprendizaje {model_name} - {dataset_name}',
        xlabel='Épocas', ylabel=loss_ylabel,
        filepath=f"{experiment_dir}/plots/training_losses_run1.png"
    )
    save_plot(
        x=df_metrics['epoch'], ys=[df_metrics['fid']],
        labels=['Puntuación FID'], colors=['green'],
        title=f'Evolución de la Calidad FID - {dataset_name}',
        xlabel='Épocas', ylabel='FID',
        filepath=f"{experiment_dir}/plots/fid_metric_run1.png",
        markers=['o']
    )
    save_plot(
        x=df_metrics['epoch'], ys=[df_metrics['is_mean']],
        labels=['Puntuación Inception (media)'], colors=['purple'],
        title=f'Evolución del Inception Score - {dataset_name}',
        xlabel='Épocas', ylabel='IS',
        filepath=f"{experiment_dir}/plots/is_metric_run1.png",
        stds=[df_metrics['is_std']], markers=['o']
    )

    # Si hay gradient penalty, graficarlo también
    if 'gp' in df_run.columns:
        save_plot(
            x=df_run['epoch'], ys=[df_run['gp']],
            labels=['Gradient Penalty'], colors=['red'],
            title=f'Evolución del Gradient Penalty - {dataset_name}',
            xlabel='Épocas', ylabel='GP',
            filepath=f"{experiment_dir}/plots/gradient_penalty_run1.png"
        )


def aggregate_runs(all_runs_data, experiment_dir, model_name, dataset_name):
    """
    Agrega los resultados de múltiples ejecuciones y genera las gráficas
    promediadas con bandas de desviación estándar.

    Detecta automáticamente:
    - Si hay 'loss_c' (crítico) o 'loss_d' (discriminador).
    - Si hay 'gp' (gradient penalty, solo WGAN-GP).

    Parameters
    ----------
    all_runs_data : list of pd.DataFrame
        Un DataFrame por cada ejecución.
    experiment_dir : str
    model_name : str
    dataset_name : str
    """
    df_all = pd.concat(all_runs_data, ignore_index=True)
    df_all.to_csv(f"{experiment_dir}/logs/metrics_all_runs.csv", index=False)

    # Detectar si es un modelo con crítico o con discriminador
    has_critic = 'loss_c' in df_all.columns
    has_gp = 'gp' in df_all.columns

    loss_disc_col = 'loss_c' if has_critic else 'loss_d'
    loss_disc_label = ('Pérdida del Crítico'
                       if has_critic else 'Pérdida del Discriminador')
    loss_ylabel = 'Pérdida Wasserstein' if has_critic else 'Pérdida'

    # Columnas de pérdida a agregar
    loss_cols = ['loss_g', loss_disc_col]
    if has_gp:
        loss_cols.append('gp')

    df_mean_loss = df_all.groupby('epoch')[loss_cols].mean().reset_index()
    df_std_loss = df_all.groupby('epoch')[loss_cols].std().reset_index()

    df_metrics = df_all.dropna(subset=['fid'])
    df_mean_metrics = (df_metrics.groupby('epoch')[['fid', 'is_mean']]
                       .mean().reset_index())
    df_std_metrics = (df_metrics.groupby('epoch')[['fid', 'is_mean']]
                      .std().reset_index())

    df_mean_final = pd.merge(df_mean_loss, df_mean_metrics,
                             on='epoch', how='left')
    df_mean_final.to_csv(f"{experiment_dir}/logs/metrics_mean.csv", index=False)

    # Curvas de pérdida promediadas (G y D/C)
    save_plot(
        x=df_mean_loss['epoch'],
        ys=[df_mean_loss['loss_g'], df_mean_loss[loss_disc_col]],
        labels=['Pérdida del Generador', loss_disc_label],
        colors=['blue', 'orange'],
        title=f'Curvas de Aprendizaje Promediadas {model_name} - {dataset_name}',
        xlabel='Épocas', ylabel=loss_ylabel,
        filepath=f"{experiment_dir}/plots/training_losses_mean.png",
        stds=[df_std_loss['loss_g'], df_std_loss[loss_disc_col]]
    )

    # FID promediado
    save_plot(
        x=df_mean_metrics['epoch'], ys=[df_mean_metrics['fid']],
        labels=['Puntuación FID'], colors=['green'],
        title=f'Evolución de la Calidad FID Promediada {model_name} - {dataset_name}',
        xlabel='Épocas', ylabel='FID',
        filepath=f"{experiment_dir}/plots/fid_metric_mean.png",
        stds=[df_std_metrics['fid']], markers=['o']
    )

    # IS promediado
    save_plot(
        x=df_mean_metrics['epoch'], ys=[df_mean_metrics['is_mean']],
        labels=['Puntuación Inception (media)'], colors=['purple'],
        title=f'Evolución del Inception Score Promediado {model_name} - {dataset_name}',
        xlabel='Épocas', ylabel='IS',
        filepath=f"{experiment_dir}/plots/is_metric_mean.png",
        stds=[df_std_metrics['is_mean']], markers=['o']
    )

    # GP promediado (solo WGAN-GP)
    if has_gp:
        save_plot(
            x=df_mean_loss['epoch'], ys=[df_mean_loss['gp']],
            labels=['Gradient Penalty'], colors=['red'],
            title=f'Evolución del Gradient Penalty Promediado - {dataset_name}',
            xlabel='Épocas', ylabel='GP',
            filepath=f"{experiment_dir}/plots/gradient_penalty_mean.png",
            stds=[df_std_loss['gp']]
        )


# ============================================================
# HELPER: formateo de logs de época
# ============================================================

def format_epoch_log(run_idx, num_runs, epoch, num_epochs, losses, metrics):
    """
    Formatea la línea de log de una época.

    Parameters
    ----------
    run_idx, num_runs, epoch, num_epochs : int
    losses : dict
        Diccionario con las pérdidas a mostrar, ej:
        {'D': 0.23, 'G': 0.45} o {'C': -1.2, 'G': 0.8, 'GP': 0.05}.
    metrics : dict
        Diccionario con 'fid', 'is_mean', 'is_std' (pueden ser NaN).

    Returns
    -------
    str
    """
    loss_str = ' | '.join(f'Pérdida {k}: {v:.4f}' for k, v in losses.items())

    fid = metrics.get('fid', np.nan)
    is_mean = metrics.get('is_mean', np.nan)
    is_std = metrics.get('is_std', np.nan)

    if not np.isnan(fid):
        metric_str = (f'FID: {fid:.2f} | '
                      f'IS: {is_mean:.2f} ± {is_std:.2f}')
    else:
        metric_str = 'FID: --- | IS: ---'

    return (f"Ejecución [{run_idx}/{num_runs}] - "
            f"Época [{epoch}/{num_epochs}] {loss_str} | {metric_str}")
