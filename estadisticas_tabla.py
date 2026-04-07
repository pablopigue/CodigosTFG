"""
Calcula las estadísticas para la tabla de métricas del TFG.
Para cada modelo muestra:
  - FID final (media ± std en la última época)
  - Mejor FID (media ± std del mínimo por run)
  - IS final (media ± std en la última época)
  - Mejor IS (media ± std del máximo por run)
  - Época media del mejor FID
  - Ratio degradación (FID final / Mejor FID)

USO:
  python estadisticas_tabla.py --base_dir ./ResultadosCodigosGeneralizacion --datasets MNIST
"""

import os
import argparse
import pandas as pd
import numpy as np

CARPETA_A_CLAVE = {
    "tfg_vanilla_gan": "Vanilla GAN",
    "tfg_dcgan":       "DCGAN",
    "tfg_dcgan_ls":    "DCGAN+LS",
    "tfg_wgan":        "WGAN",
    "tfg_wgan_conv":   "WGAN-Conv",
    "tfg_wgangp":      "WGAN-GP",
}

DATASETS_CONFIG = {
    "MNIST":        {"suffix": "mnist"},
    "FASHIONMNIST": {"suffix": "fashionmnist"},
    "SVHN":         {"suffix": "svhn"},
    "CELEBA":       {"suffix": "celeba"},
}


def procesar_dataset(base_dir, dataset_name):
    cfg = DATASETS_CONFIG[dataset_name]
    dataset_dir = os.path.join(base_dir, dataset_name)

    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] No encontrado: {dataset_dir}")
        return

    print(f"\n{'='*80}")
    print(f" {dataset_name}")
    print(f"{'='*80}")

    for prefijo, nombre in CARPETA_A_CLAVE.items():
        carpeta = os.path.join(dataset_dir, f"{prefijo}_{cfg['suffix']}")
        csv_path = os.path.join(carpeta, "logs", "metrics_all_runs.csv")

        if not os.path.exists(csv_path):
            print(f"\n  [{nombre}] No encontrado: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df_fid = df.dropna(subset=['fid'])
        n_runs = df['run'].nunique()
        last_epoch = df['epoch'].max()

        print(f"\n  {nombre} ({n_runs} runs, {last_epoch} épocas)")
        print(f"  {'-'*60}")

        # FID final
        df_last = df_fid[df_fid['epoch'] == last_epoch]
        fid_final_mean = df_last['fid'].mean()
        fid_final_std = df_last['fid'].std()
        print(f"  FID final (ép.{last_epoch}):  {fid_final_mean:.2f} ± {fid_final_std:.2f}")

        # Mejor FID por run
        best_fid_per_run = df_fid.groupby('run')['fid'].min()
        best_fid_mean = best_fid_per_run.mean()
        best_fid_std = best_fid_per_run.std()
        print(f"  Mejor FID:             {best_fid_mean:.2f} ± {best_fid_std:.2f}")

        # Época del mejor FID por run
        idx_best = df_fid.groupby('run')['fid'].idxmin()
        epocas_best = df_fid.loc[idx_best, 'epoch']
        print(f"  Época mejor FID:       {epocas_best.mean():.1f} (rango: {epocas_best.min()} - {epocas_best.max()})")

        # Ratio degradación
        ratio = fid_final_mean / best_fid_mean if best_fid_mean > 0 else float('nan')
        print(f"  Ratio final/mejor:     {ratio:.2f}x")

        # IS final
        is_final_mean = df_last['is_mean'].mean()
        is_final_std = df_last['is_mean'].std()
        print(f"  IS final (ép.{last_epoch}):   {is_final_mean:.2f} ± {is_final_std:.2f}")

        # Mejor IS por run
        best_is_per_run = df_fid.groupby('run')['is_mean'].max()
        best_is_mean = best_is_per_run.mean()
        best_is_std = best_is_per_run.std()
        print(f"  Mejor IS:              {best_is_mean:.2f} ± {best_is_std:.2f}")

        # Detalle por run
        print(f"\n  {'Run':>5s} | {'Mejor FID':>10s} | {'Ép.':>4s} | {'FID final':>10s} | {'Ratio':>6s} | {'Mejor IS':>9s} | {'IS final':>9s}")
        print(f"  {'-'*70}")
        for run in sorted(df['run'].unique()):
            dr = df_fid[df_fid['run'] == run]
            best_idx = dr['fid'].idxmin()
            best_fid = dr.loc[best_idx, 'fid']
            best_ep = dr.loc[best_idx, 'epoch']
            final_fid = dr[dr['epoch'] == last_epoch]['fid'].values[0]
            best_is = dr['is_mean'].max()
            final_is = dr[dr['epoch'] == last_epoch]['is_mean'].values[0]
            r = final_fid / best_fid if best_fid > 0 else float('nan')
            print(f"  {run:5d} | {best_fid:10.2f} | {best_ep:4.0f} | {final_fid:10.2f} | {r:5.1f}x | {best_is:9.2f} | {final_is:9.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Estadísticas detalladas para la tabla de métricas del TFG."
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
            print(f"[WARN] '{ds}' no reconocido.")

    print()


if __name__ == "__main__":
    main()
