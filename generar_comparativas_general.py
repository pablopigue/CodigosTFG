"""
Script de post-procesado comparativo para el TFG.
Lee los CSVs de cada modelo dentro de un dataset y genera:
  1. Gráfica comparativa de FID (todos los modelos superpuestos, con bandas ±1 std)
  2. Gráfica comparativa de IS (idem)
  3. Gráfica comparativa de Loss del Generador (idem)
  4. Tabla resumen CSV con FID e IS finales media y std de cada modelo
  5. Grid comparativo de imágenes generadas

USO:
  python generar_comparativas.py --base_dir /ruta/a/ResultadosCodigosGeneralizacion

ESTRUCTURA ESPERADA:
  base_dir/
    MNIST/
      tfg_vanilla_gan_mnist/logs/metrics_mean.csv
      tfg_vanilla_gan_mnist/logs/metrics_all_runs.csv
      tfg_vanilla_gan_mnist/images/epoch_80.png
      tfg_dcgan_mnist/...
      tfg_dcgan_ls_mnist/...
      tfg_wgan_mnist/...
      tfg_wgan_conv_mnist/...
      tfg_wgangp_mnist/...
    FASHIONMNIST/
      ... (misma estructura)
    SVHN/
      ...
    CELEBA/
      ...

COLUMNAS ESPERADAS EN metrics_mean.csv:
  epoch, loss_g, loss_d (o loss_c), fid, is_mean

COLUMNAS ESPERADAS EN metrics_all_runs.csv:
  epoch, loss_g, loss_d (o loss_c), fid, is_mean, is_std, run
  (más 'gp' en WGAN-GP)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

# ============================================================
# CONFIGURACIÓN: mapeo de carpetas a nombres bonitos y colores
# ============================================================

# Orden de aparición en las gráficas de más simple a más complejo
MODELOS_CONFIG = {
    "vanilla": {
        "nombre": "Vanilla GAN",
        "color": "#1f77b4",       # azul
        "linestyle": "--",
        "arq": "MLP",
        "loss_type": "BCE",
    },
    "dcgan": {
        "nombre": "DCGAN",
        "color": "#ff7f0e",       # naranja
        "linestyle": "-",
        "arq": "CNN",
        "loss_type": "BCE",
    },
    "dcgan_ls": {
        "nombre": "DCGAN+LS",
        "color": "#d62728",       # rojo
        "linestyle": "-",
        "arq": "CNN",
        "loss_type": "BCE",
    },
    "wgan": {
        "nombre": "WGAN",
        "color": "#2ca02c",       # verde
        "linestyle": "--",
        "arq": "MLP",
        "loss_type": "Wasserstein",
    },
    "wgan_conv": {
        "nombre": "WGAN-Conv",
        "color": "#9467bd",       # morado
        "linestyle": "-",
        "arq": "CNN",
        "loss_type": "Wasserstein",
    },
    "wgangp": {
        "nombre": "WGAN-GP",
        "color": "#e377c2",       # rosa
        "linestyle": "-",
        "arq": "CNN",
        "loss_type": "Wasserstein",
    },
}

# nombre de carpeta real -> clave en MODELOS_CONFIG
CARPETA_A_CLAVE = {
    "tfg_vanilla_gan":  "vanilla",
    "tfg_dcgan":        "dcgan",
    "tfg_dcgan_ls":     "dcgan_ls",
    "tfg_wgan":         "wgan",
    "tfg_wgan_conv":    "wgan_conv",
    "tfg_wgangp":       "wgangp",
}

DATASETS_CONFIG = {
    "MNIST":        {"epochs": 80,  "runs": 10, "suffix": "mnist"},
    "FASHIONMNIST": {"epochs": 80,  "runs": 10, "suffix": "fashionmnist"},
    "SVHN":         {"epochs": 150, "runs": 10, "suffix": "svhn"},
    "CELEBA":       {"epochs": 40,  "runs": 5,  "suffix": "celeba"},
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def detectar_carpeta_modelo(dataset_dir, clave_modelo, dataset_suffix):
    """Busca la carpeta del modelo dentro del directorio del dataset."""
    for prefijo, clave in CARPETA_A_CLAVE.items():
        if clave == clave_modelo:
            nombre_carpeta = f"{prefijo}_{dataset_suffix}"
            ruta = os.path.join(dataset_dir, nombre_carpeta)
            if os.path.isdir(ruta):
                return ruta
    # buscar por substring
    for d in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, d)
        if os.path.isdir(full) and clave_modelo.replace("_", "") in d.replace("_", ""):
            return full
    return None


def cargar_metrics_mean(modelo_dir):
    """Carga metrics_mean.csv de un modelo."""
    path = os.path.join(modelo_dir, "logs", "metrics_mean.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No encontrado: {path}")
        return None
    return pd.read_csv(path)


def cargar_metrics_all_runs(modelo_dir):
    """Carga metrics_all_runs.csv de un modelo."""
    path = os.path.join(modelo_dir, "logs", "metrics_all_runs.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No encontrado: {path}")
        return None
    return pd.read_csv(path)


def calcular_std_por_epoca(df_all, columna):
    """Calcula la desviación estándar de una columna agrupada por época."""
    df_valid = df_all.dropna(subset=[columna])
    if df_valid.empty:
        return None
    return df_valid.groupby('epoch')[columna].std().reset_index()


def buscar_imagen_epoca(modelo_dir, epoca):
    """Busca la imagen generada en una época concreta."""
    path = os.path.join(modelo_dir, "images", f"epoch_{epoca}.png")
    if os.path.exists(path):
        return path
    # Buscar la época más cercana disponible
    img_dir = os.path.join(modelo_dir, "images")
    if not os.path.isdir(img_dir):
        return None
    disponibles = []
    for f in os.listdir(img_dir):
        if f.startswith("epoch_") and f.endswith(".png"):
            try:
                e = int(f.replace("epoch_", "").replace(".png", ""))
                disponibles.append(e)
            except ValueError:
                pass
    if not disponibles:
        return None
    mas_cercana = min(disponibles, key=lambda x: abs(x - epoca))
    return os.path.join(img_dir, f"epoch_{mas_cercana}.png")


# ============================================================
# GENERACIÓN DE GRÁFICAS COMPARATIVAS
# ============================================================

def plot_metrica_comparativa(datos_modelos, metrica, ylabel, titulo,
                             filepath, invertir=False):
    """
    Genera una gráfica con las curvas de todos los modelos superpuestas.
    
    datos_modelos: dict {clave_modelo: (epochs, mean, std)}
    metrica: nombre de la métrica (para el label)
    invertir: si True, mejor es menor (FID)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for clave in MODELOS_CONFIG:
        if clave not in datos_modelos:
            continue
        epochs, mean, std = datos_modelos[clave]
        cfg = MODELOS_CONFIG[clave]
        
        ax.plot(epochs, mean,
                label=cfg["nombre"],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=2,
                marker='o', markersize=3)
        
        if std is not None:
            ax.fill_between(epochs,
                            mean - std,
                            mean + std,
                            color=cfg["color"],
                            alpha=0.15)
    
    ax.set_xlabel('Épocas', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(titulo, fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {filepath}")


def recortar_titulo_imagen(img_array, pct_titulo=0.10):
    """
    Recorta las imágenes generadas por matplotlib en dos pasos:
      1. Elimina padding blanco en todos los bordes
      2. Recorta un porcentaje fijo de la parte superior
    """
    if img_array.ndim == 3:
        gray = np.mean(img_array[:, :, :3], axis=2)
    else:
        gray = img_array.copy()

    h, w = gray.shape
    umbral = 0.95

    top = 0
    for row in range(h):
        if np.mean(gray[row, :] > umbral) > 0.90:
            top = row + 1
        else:
            break
    bottom = h
    for row in range(h - 1, top, -1):
        if np.mean(gray[row, :] > umbral) > 0.90:
            bottom = row
        else:
            break
    left = 0
    for col in range(w):
        if np.mean(gray[top:bottom, col] > umbral) > 0.90:
            left = col + 1
        else:
            break
    right = w
    for col in range(w - 1, left, -1):
        if np.mean(gray[top:bottom, col] > umbral) > 0.90:
            right = col
        else:
            break

    img_sin_padding = img_array[top:bottom, left:right]
    h2 = img_sin_padding.shape[0]
    corte_titulo = int(h2 * pct_titulo)
    return img_sin_padding[corte_titulo:, :]


def generar_grid_visual(modelos_imagenes, titulo, filepath, ncols=3,
                        labels_extra=None):
    """
    Grid 3×2 montado con PIL: todas las celdas idénticas, sin distorsión.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import math

    modelos_disponibles = [k for k in MODELOS_CONFIG if k in modelos_imagenes]
    n = len(modelos_disponibles)
    if n == 0:
        print("  [WARN] No hay imágenes disponibles para el grid visual.")
        return

    # 1. Cargar y recortar títulos
    imgs_raw = []
    for clave in modelos_disponibles:
        img = imread(modelos_imagenes[clave])
        img = recortar_titulo_imagen(img)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        imgs_raw.append(PILImage.fromarray(img))

    # 2. Tamaño de celda respetando aspect ratio de las imágenes
    ref_w, ref_h = imgs_raw[0].size  # PIL: (width, height)
    cell_w = 500
    cell_h = int(cell_w * ref_h / ref_w)  # proporcional
    title_h = 60      # espacio para el título de cada celda
    main_title_h = 70  # espacio para el título principal
    gap = 15           # espacio entre celdas

    nrows = math.ceil(n / ncols)

    # 3. Calcular tamaño total del canvas
    canvas_w = ncols * cell_w + (ncols - 1) * gap
    canvas_h = main_title_h + nrows * (title_h + cell_h) + (nrows - 1) * gap

    canvas = PILImage.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)

    # Fuente
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # 4. Título principal centrado
    bbox = draw.textbbox((0, 0), titulo, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, 15), titulo, fill='black', font=font_title)

    # 5. Colocar cada imagen
    for idx in range(n):
        r, c = divmod(idx, ncols)
        clave = modelos_disponibles[idx]

        x = c * (cell_w + gap)
        y = main_title_h + r * (title_h + cell_h + gap)

        # Título de celda
        nombre = MODELOS_CONFIG[clave]["nombre"]
        bbox_n = draw.textbbox((0, 0), nombre, font=font_label)
        nw = bbox_n[2] - bbox_n[0]
        draw.text((x + (cell_w - nw) // 2, y + 2), nombre,
                  fill='black', font=font_label)

        # Subtítulo si hay
        if labels_extra and clave in labels_extra:
            sub = labels_extra[clave]
            bbox_s = draw.textbbox((0, 0), sub, font=font_sub)
            sw = bbox_s[2] - bbox_s[0]
            draw.text((x + (cell_w - sw) // 2, y + 28), sub,
                      fill='#555555', font=font_sub)

        # Imagen redimensionada al tamaño de celda exacto
        img_resized = imgs_raw[idx].resize((cell_w, cell_h), PILImage.LANCZOS)
        canvas.paste(img_resized, (x, y + title_h))

    canvas.save(filepath, dpi=(150, 150))
    print(f"  Guardado: {filepath}")


def generar_tabla_resumen(resultados, dataset_name, filepath):
    """
    Genera un CSV con la tabla resumen de métricas finales.
    
    resultados: dict {clave_modelo: {fid_mean, fid_std, is_mean, is_std}}
    """
    rows = []
    for clave in MODELOS_CONFIG:
        if clave not in resultados:
            continue
        r = resultados[clave]
        cfg = MODELOS_CONFIG[clave]
        rows.append({
            "Modelo": cfg["nombre"],
            "Arquitectura": cfg["arq"],
            "Pérdida": cfg["loss_type"],
            "FID (media)": f"{r['fid_mean']:.2f}",
            "FID (std)": f"{r['fid_std']:.2f}",
            "IS (media)": f"{r['is_mean']:.2f}",
            "IS (std)": f"{r['is_std']:.2f}",
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Guardado: {filepath}")
    print(f"\n  Tabla resumen {dataset_name}:")
    print(df.to_string(index=False))
    print()


# ============================================================
# BUCLE PRINCIPAL POR DATASET
# ============================================================

def procesar_dataset(base_dir, dataset_name):
    """Procesa un dataset completo: carga datos, genera gráficas y tablas."""
    
    cfg_ds = DATASETS_CONFIG[dataset_name]
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    if not os.path.isdir(dataset_dir):
        print(f"\n[SKIP] Directorio no encontrado: {dataset_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f" Procesando: {dataset_name}")
    print(f" Directorio: {dataset_dir}")
    print(f" Épocas: {cfg_ds['epochs']} | Runs: {cfg_ds['runs']}")
    print(f"{'='*60}")
    
    # Directorio de salida para las gráficas comparativas
    out_dir = os.path.join(dataset_dir, "_comparativas")
    os.makedirs(out_dir, exist_ok=True)
    
    # Recopilar datos de todos los modelos
    datos_fid = {}
    datos_is = {}
    datos_loss_g = {}
    resultados_finales = {}
    imagenes_finales = {}
    
    for clave in MODELOS_CONFIG:
        modelo_dir = detectar_carpeta_modelo(
            dataset_dir, clave, cfg_ds["suffix"]
        )
        if modelo_dir is None:
            print(f"  [SKIP] Modelo '{clave}' no encontrado en {dataset_dir}")
            continue
        
        print(f"\n  Cargando: {os.path.basename(modelo_dir)}")
        
        # Cargar metrics_mean.csv
        df_mean = cargar_metrics_mean(modelo_dir)
        if df_mean is None:
            continue
        
        # Cargar metrics_all_runs.csv para std
        df_all = cargar_metrics_all_runs(modelo_dir)
        
        # FID
        df_fid = df_mean.dropna(subset=['fid'])
        if not df_fid.empty:
            fid_std = None
            if df_all is not None:
                std_df = calcular_std_por_epoca(df_all, 'fid')
                if std_df is not None:
                    # Merge para alinear épocas
                    merged = df_fid.merge(std_df, on='epoch',
                                          suffixes=('_mean', '_std'))
                    fid_std = merged['fid_std'].values
                    datos_fid[clave] = (
                        merged['epoch'].values,
                        merged['fid_mean'].values,
                        fid_std
                    )
                else:
                    datos_fid[clave] = (
                        df_fid['epoch'].values,
                        df_fid['fid'].values,
                        None
                    )
            else:
                datos_fid[clave] = (
                    df_fid['epoch'].values,
                    df_fid['fid'].values,
                    None
                )
        
        # IS
        df_is = df_mean.dropna(subset=['is_mean'])
        if not df_is.empty:
            is_std = None
            if df_all is not None:
                std_df = calcular_std_por_epoca(df_all, 'is_mean')
                if std_df is not None:
                    merged = df_is.merge(std_df, on='epoch',
                                         suffixes=('_mean', '_std'))
                    is_std = merged['is_mean_std'].values
                    datos_is[clave] = (
                        merged['epoch'].values,
                        merged['is_mean_mean'].values,
                        is_std
                    )
                else:
                    datos_is[clave] = (
                        df_is['epoch'].values,
                        df_is['is_mean'].values,
                        None
                    )
            else:
                datos_is[clave] = (
                    df_is['epoch'].values,
                    df_is['is_mean'].values,
                    None
                )
        
        # Loss G
        loss_g_col = 'loss_g'
        if loss_g_col in df_mean.columns:
            loss_g_std = None
            if df_all is not None and loss_g_col in df_all.columns:
                std_df = df_all.groupby('epoch')[loss_g_col].std().reset_index()
                std_df.columns = ['epoch', 'loss_g_std']
                merged = df_mean[['epoch', loss_g_col]].merge(
                    std_df, on='epoch'
                )
                datos_loss_g[clave] = (
                    merged['epoch'].values,
                    merged[loss_g_col].values,
                    merged['loss_g_std'].values
                )
            else:
                datos_loss_g[clave] = (
                    df_mean['epoch'].values,
                    df_mean[loss_g_col].values,
                    None
                )
        
        # Métricas finales (última época con FID calculado)
        if df_all is not None:
            df_final = df_all.dropna(subset=['fid'])
            if not df_final.empty:
                ultima_epoca = df_final['epoch'].max()
                df_ult = df_final[df_final['epoch'] == ultima_epoca]
                resultados_finales[clave] = {
                    'fid_mean': df_ult['fid'].mean(),
                    'fid_std': df_ult['fid'].std(),
                    'is_mean': df_ult['is_mean'].mean(),
                    'is_std': df_ult['is_mean'].std(),
                }
                print(f"    Época final ({ultima_epoca}): "
                      f"FID={resultados_finales[clave]['fid_mean']:.2f}"
                      f"±{resultados_finales[clave]['fid_std']:.2f} | "
                      f"IS={resultados_finales[clave]['is_mean']:.2f}"
                      f"±{resultados_finales[clave]['is_std']:.2f}")
        
        # Imagen de la última época
        img_path = buscar_imagen_epoca(modelo_dir, cfg_ds["epochs"])
        if img_path:
            imagenes_finales[clave] = img_path
    
    # Generar gráficas
    print(f"\n  Generando gráficas comparativas...")
    
    if datos_fid:
        plot_metrica_comparativa(
            datos_fid, "FID", "",
            f"Evolución del FID — {dataset_name}",
            os.path.join(out_dir, "comparativa_fid.png"),
            invertir=True
        )
    
    if datos_is:
        plot_metrica_comparativa(
            datos_is, "IS", "",
            f"Evolución del IS — {dataset_name}",
            os.path.join(out_dir, "comparativa_is.png"),
            invertir=False
        )
    
    if datos_loss_g:
        plot_metrica_comparativa(
            datos_loss_g, "Loss G", "Pérdida del Generador",
            f"Pérdida del Generador — {dataset_name} ({cfg_ds['runs']} runs)",
            os.path.join(out_dir, "comparativa_loss_g.png")
        )
    
    if resultados_finales:
        generar_tabla_resumen(
            resultados_finales, dataset_name,
            os.path.join(out_dir, "tabla_resumen.csv")
        )
    
    if imagenes_finales:
        generar_grid_visual(
            imagenes_finales,
            f"Muestras generadas — {dataset_name} (época final)",
            os.path.join(out_dir, "grid_visual_comparativa.png")
        )


def generar_tabla_global(base_dir):
    """
    Genera una tabla resumen global: FID final de cada modelo × cada dataset.
    Lee las tablas individuales ya generadas.
    """
    print(f"\n{'='*60}")
    print(f" Generando tabla resumen GLOBAL")
    print(f"{'='*60}")
    
    all_data = {}
    for ds_name in DATASETS_CONFIG:
        csv_path = os.path.join(base_dir, ds_name, "_comparativas",
                                "tabla_resumen.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                modelo = row['Modelo']
                if modelo not in all_data:
                    all_data[modelo] = {}
                all_data[modelo][ds_name] = (
                    f"{row['FID (media)']}±{row['FID (std)']}"
                )
    
    if not all_data:
        print("  [WARN] No se encontraron tablas individuales.")
        return
    
    # Construir DataFrame global
    rows = []
    for clave in MODELOS_CONFIG:
        nombre = MODELOS_CONFIG[clave]["nombre"]
        if nombre in all_data:
            row = {"Modelo": nombre, "Arq.": MODELOS_CONFIG[clave]["arq"]}
            for ds_name in DATASETS_CONFIG:
                row[ds_name] = all_data[nombre].get(ds_name, "—")
            rows.append(row)
    
    df_global = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, "tabla_resumen_global.csv")
    df_global.to_csv(out_path, index=False)
    print(f"\n  Tabla global guardada: {out_path}")
    print(df_global.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera gráficas comparativas y tablas resumen "
                    "para el análisis de resultados del TFG."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./ResultadosCodigosGeneralizacion",
        help="Ruta al directorio raíz con las carpetas MNIST, FASHIONMNIST, etc."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets a procesar (ej: MNIST SVHN). Si no se indica, procesa todos."
    )
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.base_dir)
    print(f"Directorio base: {base_dir}")
    
    datasets = args.datasets or list(DATASETS_CONFIG.keys())
    
    for ds in datasets:
        if ds.upper() in DATASETS_CONFIG:
            procesar_dataset(base_dir, ds.upper())
        else:
            print(f"\n[WARN] Dataset '{ds}' no reconocido. "
                  f"Opciones: {list(DATASETS_CONFIG.keys())}")
    
    # Tabla resumen global
    generar_tabla_global(base_dir)
    
    print(f"\n{'='*60}")
    print(f" Procesado completo.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
