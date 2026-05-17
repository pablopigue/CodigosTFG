import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CHANNELS = 3
Z_DIM    = 100

BG        = "#F8F8F6"
C_CONV    = "#378ADD"
C_CONVT   = "#7F77DD"
C_BN      = "#1D9E75"
C_LRELU   = "#BA7517"
C_RELU    = "#1D9E75"
C_TANH    = "#7F77DD"
C_SIGMOID = "#D85A30"
C_INPUT   = "#888780"
C_LABEL   = "#333333"
C_ARROW   = "#888888"

discriminator_layers = [
    dict(spatial=32, channels=CHANNELS, label=f"Input\n32×32×{CHANNELS}", color=C_INPUT),
    dict(spatial=16, channels=64,  label="Conv2d\n→16×16×64",  color=C_CONV),
    dict(label="LeakyReLU", color=C_LRELU),
    dict(spatial=8,  channels=128, label="Conv2d\n→8×8×128",   color=C_CONV),
    dict(label="Norm", color=C_BN),
    dict(label="LeakyReLU", color=C_LRELU),
    dict(spatial=4,  channels=256, label="Conv2d\n→4×4×256",   color=C_CONV),
    dict(label="Norm", color=C_BN),
    dict(label="LeakyReLU", color=C_LRELU),
    dict(spatial=1,  channels=1,   label="Conv2d\n→1×1×1",     color=C_CONV),
    dict(label="Sigmoid", color=C_SIGMOID),
]

generator_layers = [
    dict(spatial=1,  channels=Z_DIM,    label=f"Noise\n1×1\n×{Z_DIM}",  color=C_INPUT),
    dict(spatial=4,  channels=256,      label="ConvT\n→4×4×256",      color=C_CONVT),
    dict(label="BatchNorm", color=C_BN),
    dict(label="ReLU", color=C_RELU),
    dict(spatial=8,  channels=128,      label="ConvT\n→8×8×128",      color=C_CONVT),
    dict(label="BatchNorm", color=C_BN),
    dict(label="ReLU", color=C_RELU),
    dict(spatial=16, channels=64,       label="ConvT\n→16×16×64",     color=C_CONVT),
    dict(label="BatchNorm", color=C_BN),
    dict(label="ReLU", color=C_RELU),
    dict(spatial=32, channels=CHANNELS, label=f"ConvT\n→32×32×{CHANNELS}", color=C_CONVT),
    dict(label="Tanh", color=C_TANH),
]


def spatial_to_w(s, min_w=0.12, max_w=0.55, ref=32):
    return min_w + (max_w - min_w) * (np.log1p(s) / np.log1p(ref))


def channels_to_h(c, min_h=0.4, max_h=3.6, ref=256):
    return min_h + (max_h - min_h) * (np.log1p(c) / np.log1p(ref))


def is_activation_layer(layer):
    """Devuelve True si la capa no tiene información espacial (es activación/norm)."""
    return "spatial" not in layer and "channels" not in layer


def draw_volume(ax, x, y_center, w, h, color, label, label_position, zorder=3):
    """
    Dibuja un bloque 3D y su etiqueta alineada.
    """
    iso_dx = w * 0.45
    iso_dy = w * 0.30
    y = y_center - h / 2

    # Cara frontal
    front = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        fc=color, ec="white", lw=1.0, zorder=zorder
    )
    ax.add_patch(front)

    # Cara superior
    tx = [x, x+w, x+w+iso_dx, x+iso_dx, x]
    ty = [y+h, y+h, y+h+iso_dy, y+h+iso_dy, y+h]
    ax.fill(tx, ty, color=color, alpha=0.45, zorder=zorder-1)
    ax.plot(tx, ty, color="white", lw=0.7, zorder=zorder+1)

    # Cara lateral derecha
    sx = [x+w, x+w+iso_dx, x+w+iso_dx, x+w, x+w]
    sy = [y,   y+iso_dy,   y+h+iso_dy, y+h,  y]
    ax.fill(sx, sy, color=color, alpha=0.25, zorder=zorder-1)
    ax.plot(sx, sy, color="white", lw=0.7, zorder=zorder+1)

    # Renderizado de etiquetas con línea de base fija y conectores
    fontsize = 6.5 if "\n" in label else 7.0
    text_x = x + w/2 + iso_dx/2

    if label_position == "main":
        # Carril inferior para las capas físicas (Conv, Input)
        label_y = -0.1
        ax.text(
            text_x, label_y, label,
            ha="center", va="top",
            fontsize=fontsize, color=color, fontweight="bold",
            linespacing=1.35, zorder=zorder+2
        )
        # Línea punteada que ancla el bloque a su texto
        ax.plot(
            [text_x, text_x],
            [y - 0.05, label_y + 0.1],
            color=color, lw=1.2, ls=":", alpha=0.5, zorder=zorder+1
        )
        
    elif label_position == "activation":
        # Carril superior para activaciones y normalizaciones
        label_y = 5.5
        ax.text(
            text_x, label_y, label,
            ha="center", va="center",
            fontsize=6.5, color=color, fontweight="bold",
            linespacing=1.3, zorder=zorder+2,
            bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=color, lw=1.0, alpha=0.95)
        )
        # Línea punteada que ancla el bloque a su texto
        ax.plot(
            [text_x, text_x],
            [y + h + iso_dy + 0.05, label_y - 0.25],
            color=color, lw=1.2, ls=":", alpha=0.5, zorder=zorder+1
        )

    return x + w + iso_dx


def draw_network(ax, layers, title, gap=0.48):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    CENTER_Y = 2.6
    x = 0.3

    last_spatial = 32
    last_channels = 3

    # Asignamos estrictamente "main" (abajo) o "activation" (arriba)
    label_positions = []
    for layer in layers:
        if is_activation_layer(layer):
            label_positions.append("activation")
        else:
            label_positions.append("main")

    for i, layer in enumerate(layers):
        current_spatial = layer.get("spatial", last_spatial)
        current_channels = layer.get("channels", last_channels)
        last_spatial = current_spatial
        last_channels = current_channels

        w = spatial_to_w(current_spatial)
        h = channels_to_h(current_channels)

        right_edge = draw_volume(
            ax, x, CENTER_Y, w, h,
            layer["color"], layer["label"],
            label_position=label_positions[i]
        )

        # Flecha de conexión
        if i < len(layers) - 1:
            arr_x0 = right_edge + 0.04
            arr_x1 = right_edge + gap - 0.10
            ax.annotate(
                "", xy=(arr_x1, CENTER_Y),
                xytext=(arr_x0, CENTER_Y),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.1),
                zorder=2
            )
            x = right_edge + gap
        else:
            x = right_edge

    ax.set_xlim(-0.1, x + 0.3)
    ax.set_ylim(-1.0, 6.2) # Ampliado para dar espacio a los nuevos carriles fijos


def save_figure(layers, title, filename, figw=24): # Aumentado un poco el ancho
    fig, ax = plt.subplots(figsize=(figw, 4.5), facecolor=BG)
    draw_network(ax, layers, title)

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {filename}")


save_figure(generator_layers,     "Generador-Conv",     "conv_generator.png",     figw=25)
save_figure(discriminator_layers, "Discriminador-Conv", "conv_discriminator.png", figw=25)