"""
analyze_benchmark.py

A partir de benchmark_results.csv y benchmark_results.json, genera:
    1. Tabla LaTeX con tiempos medios por (modelo, dataset) y ratio vs DCGAN.
    2. Gráfica de barras agrupadas: tiempo por época por modelo en cada dataset.
    3. Gráfica de ms por batch: normaliza tamaño de batch para comparar
       el coste real de una actualización entre modelos.
    4. Gráfica de escalado: cómo crece el tiempo del modelo al pasar
       de 32x32 (3 datasets) a 64x64 (CelebA).

Uso:
    python analyze_benchmark.py

Salida:
    - tabla_tiempos.tex
    - grafica_tiempos_por_epoca.png
    - grafica_ms_por_batch.png
    - grafica_escalado_resolucion.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('agg')

# ============================================================
# CARGA
# ============================================================
df = pd.read_csv("benchmark_results.csv")
with open("benchmark_results.json") as f:
    raw = json.load(f)

# Orden canónico de modelos y datasets (para gráficas consistentes)
MODEL_ORDER = ["Vanilla GAN", "DCGAN", "DCGAN+LS", "WGAN", "WGAN-Conv", "WGAN-GP"]
DATASET_ORDER = ["MNIST", "FashionMNIST", "SVHN", "CelebA"]

df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
df = df.sort_values(["dataset", "model"]).reset_index(drop=True)


# ============================================================
# TABLA LATEX
# ============================================================
def format_time(mean, std):
    """Formato con comas decimales (convención española)."""
    return (f"{mean:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            + " $\\pm$ "
            + f"{std:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))


# Pivoteo: filas = modelo, columnas = dataset
pivot_mean = df.pivot(index="model", columns="dataset",
                     values="mean_time_per_epoch_s")
pivot_std = df.pivot(index="model", columns="dataset",
                    values="std_time_per_epoch_s")
pivot_mean = pivot_mean.reindex(MODEL_ORDER)[DATASET_ORDER]
pivot_std = pivot_std.reindex(MODEL_ORDER)[DATASET_ORDER]

# Ratio vs DCGAN en cada dataset
ratios = pivot_mean.div(pivot_mean.loc["DCGAN"])

# Construir el cuerpo de la tabla LaTeX
lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(r"\caption{Tiempo medio por época (en segundos) de cada modelo "
             r"en cada dataset, medido sobre 5 épocas cronometradas tras una "
             r"época de warmup. Entre paréntesis, ratio respecto al tiempo "
             r"de DCGAN en el mismo dataset.}")
lines.append(r"\label{tab:benchmark_tiempos}")
lines.append(r"\resizebox{\textwidth}{!}{")
lines.append(r"\begin{tabular}{llcccc}")
lines.append(r"\hline")
lines.append(r"\textbf{Modelo} & \textbf{Arq.} & "
             + " & ".join(fr"\textbf{{{d}}}" for d in DATASET_ORDER)
             + r" \\")
lines.append(r"\hline\hline")

arch_map = {m: "MLP" if m in ("Vanilla GAN", "WGAN") else "CNN"
            for m in MODEL_ORDER}

for model in MODEL_ORDER:
    cells = []
    for ds in DATASET_ORDER:
        mean = pivot_mean.loc[model, ds]
        std = pivot_std.loc[model, ds]
        ratio = ratios.loc[model, ds]
        cell = (f"{format_time(mean, std)} "
                f"($\\times {ratio:.2f}$)".replace(".", ","))
        cells.append(cell)
    lines.append(f"{model} & {arch_map[model]} & " + " & ".join(cells) + r" \\")

lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"}")
lines.append(r"\end{table}")

tabla_latex = "\n".join(lines)
with open("tabla_tiempos.tex", "w") as f:
    f.write(tabla_latex)

print("Tabla LaTeX generada en tabla_tiempos.tex")
print("-" * 60)
print(tabla_latex)
print("-" * 60)


# ============================================================
# GRÁFICA 1: TIEMPO POR ÉPOCA
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(DATASET_ORDER))
n_models = len(MODEL_ORDER)
bar_width = 0.8 / n_models

colors = {
    "Vanilla GAN": "#4C72B0",
    "DCGAN":       "#DD8452",
    "DCGAN+LS":    "#C44E52",
    "WGAN":        "#8172B3",
    "WGAN-Conv":   "#55A868",
    "WGAN-GP":     "#CCB974",
}

for i, model in enumerate(MODEL_ORDER):
    means = [pivot_mean.loc[model, ds] for ds in DATASET_ORDER]
    stds = [pivot_std.loc[model, ds] for ds in DATASET_ORDER]
    offset = (i - n_models/2) * bar_width + bar_width/2
    ax.bar(x + offset, means, bar_width, yerr=stds,
           label=model, color=colors[model], capsize=3,
           edgecolor='black', linewidth=0.5)

ax.set_xlabel("Dataset")
ax.set_ylabel("Tiempo medio por época (s)")
ax.set_title("Coste computacional por época de cada modelo y dataset")
ax.set_xticks(x)
ax.set_xticklabels(DATASET_ORDER)
ax.legend(loc='upper left', ncol=2)
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("grafica_tiempos_por_epoca.png", dpi=150)
plt.close()
print("Gráfica grafica_tiempos_por_epoca.png generada.")


# Versión en escala logarítmica
fig, ax = plt.subplots(figsize=(12, 6))
for i, model in enumerate(MODEL_ORDER):
    means = [pivot_mean.loc[model, ds] for ds in DATASET_ORDER]
    stds = [pivot_std.loc[model, ds] for ds in DATASET_ORDER]
    offset = (i - n_models/2) * bar_width + bar_width/2
    ax.bar(x + offset, means, bar_width, yerr=stds,
           label=model, color=colors[model], capsize=3,
           edgecolor='black', linewidth=0.5)
ax.set_xlabel("Dataset")
ax.set_ylabel("Tiempo medio por época (s) — escala log")
ax.set_yscale('log')
ax.set_title("Coste computacional por época (escala logarítmica)")
ax.set_xticks(x)
ax.set_xticklabels(DATASET_ORDER)
ax.legend(loc='upper left', ncol=2)
ax.grid(True, axis='y', alpha=0.3, which='both')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("grafica_tiempos_por_epoca_log.png", dpi=150)
plt.close()
print("Gráfica grafica_tiempos_por_epoca_log.png generada.")


# ============================================================
# GRÁFICA 2: MILISEGUNDOS POR BATCH
# Normaliza el batch_size: comparable entre DCGAN (bs=128) y WGAN (bs=64).
# ============================================================
pivot_ms = df.pivot(index="model", columns="dataset",
                    values="ms_per_batch").reindex(MODEL_ORDER)[DATASET_ORDER]

fig, ax = plt.subplots(figsize=(12, 6))
for i, model in enumerate(MODEL_ORDER):
    vals = [pivot_ms.loc[model, ds] for ds in DATASET_ORDER]
    offset = (i - n_models/2) * bar_width + bar_width/2
    ax.bar(x + offset, vals, bar_width,
           label=model, color=colors[model],
           edgecolor='black', linewidth=0.5)

ax.set_xlabel("Dataset")
ax.set_ylabel("Tiempo medio por batch (ms)")
ax.set_title("Coste por batch (actualización completa: crítico/discriminador + generador)")
ax.set_xticks(x)
ax.set_xticklabels(DATASET_ORDER)
ax.legend(loc='upper left', ncol=2)
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("grafica_ms_por_batch.png", dpi=150)
plt.close()
print("Gráfica grafica_ms_por_batch.png generada.")


# ============================================================
# GRÁFICA 3: ESCALADO CON LA RESOLUCIÓN
# Compara 32x32 (media de MNIST/FashionMNIST/SVHN) con 64x64 (CelebA)
# ============================================================
df_32 = df[df["img_size"] == 32].groupby("model", observed=True)[
    "mean_time_per_epoch_s"].mean().reindex(MODEL_ORDER)
df_64 = df[df["img_size"] == 64].groupby("model", observed=True)[
    "mean_time_per_epoch_s"].mean().reindex(MODEL_ORDER)

fig, ax = plt.subplots(figsize=(10, 6))
x2 = np.arange(len(MODEL_ORDER))
w = 0.4
ax.bar(x2 - w/2, df_32.values, w, label="$32\\times32$ (media MNIST/FashionMNIST/SVHN)",
       color="#4C72B0", edgecolor='black', linewidth=0.5)
ax.bar(x2 + w/2, df_64.values, w, label="$64\\times64$ (CelebA)",
       color="#DD8452", edgecolor='black', linewidth=0.5)
ax.set_xticks(x2)
ax.set_xticklabels(MODEL_ORDER, rotation=20, ha='right')
ax.set_ylabel("Tiempo medio por época (s)")
ax.set_title("Escalado del coste con la resolución")
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("grafica_escalado_resolucion.png", dpi=150)
plt.close()
print("Gráfica grafica_escalado_resolucion.png generada.")


# ============================================================
# RESUMEN EN STDOUT
# ============================================================
print("\n" + "=" * 60)
print("RESUMEN DE TIEMPOS")
print("=" * 60)
print(f"\nTabla pivoteada (s/época):\n")
print(pivot_mean.round(2).to_string())
print(f"\nRatios respecto a DCGAN:\n")
print(ratios.round(2).to_string())
