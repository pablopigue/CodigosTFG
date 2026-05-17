"""
Grid visual comparativo: imagen de la época con MEJOR FID de la run 1.
Disposición 3x2, todas las imágenes al mismo tamaño.

USO:
  python grid_mejor_fid.py --base_dir ./ResultadosCodigosGeneralizacion
  python grid_mejor_fid.py --base_dir ./ResultadosCodigosGeneralizacion --datasets MNIST SVHN
"""

import os
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image as PILImage

# ============================================================
# CONFIGURACIÓN
# ============================================================

MODELOS_CONFIG = {
    "vanilla":   {"nombre": "Vanilla GAN", "arq": "MLP"},
    "dcgan":     {"nombre": "DCGAN",       "arq": "CNN"},
    "dcgan_ls":  {"nombre": "DCGAN+LS",    "arq": "CNN"},
    "wgan":      {"nombre": "WGAN-MLP",    "arq": "MLP"},
    "wgan_conv": {"nombre": "WGAN-Conv",   "arq": "CNN"},
    "wgangp":    {"nombre": "WGAN-GP",     "arq": "CNN"},
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
    "MNIST":        {"epochs": 80,  "suffix": "mnist"},
    "FASHIONMNIST": {"epochs": 80,  "suffix": "fashionmnist"},
    "SVHN":         {"epochs": 150, "suffix": "svhn"},
    "CELEBA":       {"epochs": 40,  "suffix": "celeba"},
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


def obtener_mejor_fid_run1(modelo_dir):
    """Mejor FID de la run 1."""
    for nombre in ["metrics_all_runs.csv", "metrics_run1.csv"]:
        csv_path = os.path.join(modelo_dir, "logs", nombre)
        if os.path.exists(csv_path):
            break
    else:
        return None, None

    df = pd.read_csv(csv_path)
    if 'run' in df.columns:
        df = df[df['run'] == 1]
    df = df.dropna(subset=['fid'])
    if df.empty:
        return None, None

    idx_min = df['fid'].idxmin()
    return int(df.loc[idx_min, 'epoch']), df.loc[idx_min, 'fid']


def buscar_imagen_mas_cercana(modelo_dir, epoca_objetivo):
    img_dir = os.path.join(modelo_dir, "images")
    if not os.path.isdir(img_dir):
        return None, None

    disponibles = []
    for f in os.listdir(img_dir):
        if f.startswith("epoch_") and f.endswith(".png"):
            try:
                disponibles.append(int(f.replace("epoch_", "").replace(".png", "")))
            except ValueError:
                pass
    if not disponibles:
        return None, None

    mas_cercana = min(disponibles, key=lambda x: abs(x - epoca_objetivo))
    return os.path.join(img_dir, f"epoch_{mas_cercana}.png"), mas_cercana


def recortar_titulo_imagen(img_array, pct_titulo=0.10):
    """
    Recorta en dos pasos:
      1. Padding blanco de los 4 bordes
      2. Porcentaje fijo de la parte superior
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


def generar_grid(modelos_info, titulo, filepath, ncols=3):
    """
    Grid 3×2 montado con PIL: todas las celdas idénticas, sin distorsión.
    modelos_info: dict {clave: {ruta_img, epoca_img, fid}}
    """
    from PIL import ImageDraw, ImageFont

    modelos_disponibles = [k for k in MODELOS_CONFIG if k in modelos_info]
    n = len(modelos_disponibles)
    if n == 0:
        print("  [WARN] No hay imágenes disponibles.")
        return

    # 1. Cargar y recortar títulos
    imgs = []
    for clave in modelos_disponibles:
        img = imread(modelos_info[clave]['ruta_img'])
        img = recortar_titulo_imagen(img)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        imgs.append(PILImage.fromarray(img))

    # 2. Tamaño de celda respetando aspect ratio de las imágenes
    #    Usamos el aspect ratio de la primera imagen como referencia
    ref_w, ref_h = imgs[0].size
    cell_w = 500
    cell_h = int(cell_w * ref_h / ref_w) 
    title_h = 70       
    main_title_h = 70
    gap = 15

    nrows = math.ceil(n / ncols)

    canvas_w = ncols * cell_w + (ncols - 1) * gap
    canvas_h = main_title_h + nrows * (title_h + cell_h) + (nrows - 1) * gap

    canvas = PILImage.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Título principal
    bbox = draw.textbbox((0, 0), titulo, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, 15), titulo, fill='black', font=font_title)

    # Celdas
    for idx in range(n):
        r, c = divmod(idx, ncols)
        clave = modelos_disponibles[idx]
        info = modelos_info[clave]

        x = c * (cell_w + gap)
        y = main_title_h + r * (title_h + cell_h + gap)

        # Nombre del modelo
        nombre = MODELOS_CONFIG[clave]["nombre"]
        bbox_n = draw.textbbox((0, 0), nombre, font=font_label)
        nw = bbox_n[2] - bbox_n[0]
        draw.text((x + (cell_w - nw) // 2, y + 2), nombre,
                  fill='black', font=font_label)

        # Línea de época + FID
        sub = f"Ép. {info['epoca_img']}  |  FID: {info['fid']:.1f}"
        bbox_s = draw.textbbox((0, 0), sub, font=font_sub)
        sw = bbox_s[2] - bbox_s[0]
        draw.text((x + (cell_w - sw) // 2, y + 30), sub,
                  fill='#555555', font=font_sub)

        # Imagen
        img_resized = imgs[idx].resize((cell_w, cell_h), PILImage.LANCZOS)
        canvas.paste(img_resized, (x, y + title_h))

    canvas.save(filepath, dpi=(150, 150))
    print(f"  Guardado: {filepath}")


# ============================================================
# PROCESAMIENTO
# ============================================================

def procesar_dataset(base_dir, dataset_name):
    cfg_ds = DATASETS_CONFIG[dataset_name]
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        print(f"\n[SKIP] No encontrado: {dataset_dir}")
        return

    print(f"\n{'='*60}")
    print(f" Procesando: {dataset_name}")
    print(f"{'='*60}")

    out_dir = os.path.join(dataset_dir, "_comparativas")
    os.makedirs(out_dir, exist_ok=True)

    modelos_info = {}

    for clave in MODELOS_CONFIG:
        modelo_dir = detectar_carpeta_modelo(dataset_dir, clave, cfg_ds["suffix"])
        if modelo_dir is None:
            print(f"  [SKIP] {clave} no encontrado")
            continue

        mejor_epoca, mejor_fid = obtener_mejor_fid_run1(modelo_dir)
        if mejor_epoca is None:
            print(f"  [SKIP] {clave}: no hay datos de FID para run 1")
            continue

        ruta_img, epoca_img = buscar_imagen_mas_cercana(modelo_dir, mejor_epoca)
        if ruta_img is None:
            print(f"  [SKIP] {clave}: no hay imágenes")
            continue

        modelos_info[clave] = {
            'ruta_img': ruta_img,
            'epoca_img': epoca_img,
            'fid': mejor_fid,
        }

        nota = f" (img: ép. {epoca_img})" if epoca_img != mejor_epoca else ""
        print(f"  {MODELOS_CONFIG[clave]['nombre']:15s} → "
              f"Mejor FID run 1: {mejor_fid:.2f} en época {mejor_epoca}{nota}")

    if modelos_info:
        generar_grid(
            modelos_info,
            f"Resultado con mejor FID por modelo — {dataset_name}",
            os.path.join(out_dir, "grid_mejor_fid.png")
        )


def main():
    parser = argparse.ArgumentParser(
        description="Grid visual 3x2: época con mejor FID de la run 1."
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
