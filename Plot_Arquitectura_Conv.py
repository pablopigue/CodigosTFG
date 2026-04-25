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
    """Returns True if layer has no spatial/channels info (activation/norm layer)."""
    return "spatial" not in layer and "channels" not in layer


def draw_volume(ax, x, y_center, w, h, color, label, label_position="center", zorder=3):
    """
    Draws a 3D block (feature map).
    label_position: "center", "above", "below"
    """
    iso_dx = w * 0.45
    iso_dy = w * 0.30
    y = y_center - h / 2

    # Front face
    front = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        fc=color, ec="white", lw=1.0, zorder=zorder
    )
    ax.add_patch(front)

    # Top face
    tx = [x, x+w, x+w+iso_dx, x+iso_dx, x]
    ty = [y+h, y+h, y+h+iso_dy, y+h+iso_dy, y+h]
    ax.fill(tx, ty, color=color, alpha=0.45, zorder=zorder-1)
    ax.plot(tx, ty, color="white", lw=0.7, zorder=zorder+1)

    # Right face
    sx = [x+w, x+w+iso_dx, x+w+iso_dx, x+w, x+w]
    sy = [y,   y+iso_dy,   y+h+iso_dy, y+h,  y]
    ax.fill(sx, sy, color=color, alpha=0.25, zorder=zorder-1)
    ax.plot(sx, sy, color="white", lw=0.7, zorder=zorder+1)

    # Label rendering
    fontsize = 6.5 if "\n" in label else 7.0

    if label_position == "center":
        # Label inside the block (only for wide enough blocks)
        ax.text(
            x + w/2, y_center, label,
            ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold",
            linespacing=1.35, zorder=zorder+2,
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")]
        )
    elif label_position == "above":
        # Label above the block on a connector line
        label_y = y + h + iso_dy + 0.35
        ax.text(
            x + w/2 + iso_dx/2, label_y, label,
            ha="center", va="bottom",
            fontsize=6.5, color=color, fontweight="bold",
            linespacing=1.3, zorder=zorder+2,
            bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec=color, lw=0.8, alpha=0.92)
        )
        # Connector line from top of block to label box
        ax.plot(
            [x + w/2 + iso_dx/2, x + w/2 + iso_dx/2],
            [y + h + iso_dy + 0.04, label_y - 0.02],
            color=color, lw=0.8, ls="--", alpha=0.7, zorder=zorder+1
        )
    elif label_position == "below":
        # Label below the block
        label_y = y - 0.35
        ax.text(
            x + w/2, label_y, label,
            ha="center", va="top",
            fontsize=6.5, color=color, fontweight="bold",
            linespacing=1.3, zorder=zorder+2,
            bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec=color, lw=0.8, alpha=0.92)
        )
        # Connector line from bottom of block to label box
        ax.plot(
            [x + w/2, x + w/2],
            [y - 0.04, label_y + 0.08],
            color=color, lw=0.8, ls="--", alpha=0.7, zorder=zorder+1
        )

    return x + w + iso_dx  # right edge including perspective


def draw_network(ax, layers, title, gap=0.28):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    CENTER_Y = 2.8
    x = 0.3

    last_spatial = 32
    last_channels = 3

    # Pre-pass: decide label positions for activation layers to alternate above/below
    # Track the sequence of activation layers between conv layers
    activation_counter = 0

    label_positions = []
    act_idx = 0  # alternating counter for activations between conv blocks

    for i, layer in enumerate(layers):
        if is_activation_layer(layer):
            # Alternate: even -> above, odd -> below
            label_positions.append("above" if act_idx % 2 == 0 else "below")
            act_idx += 1
        else:
            label_positions.append("center")
            act_idx = 0  # reset after each conv layer

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

        # Dimension annotation below block (only for explicit spatial layers)
        if "spatial" in layer:
            ax.text(
                x + w/2, CENTER_Y - h/2 - 0.55,
                f"{layer['spatial']}×{layer['spatial']}\n×{layer['channels']}",
                ha="center", va="top", fontsize=5.5, color=C_LABEL,
                linespacing=1.3
            )

        # Arrow to next block
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
    ax.set_ylim(-0.5, 6.2)

    pass



def save_figure(layers, title, filename, figw=22):
    fig, ax = plt.subplots(figsize=(figw, 5.0), facecolor=BG)
    draw_network(ax, layers, title)

    # suptitle centrado + nota en cursiva
    fig.suptitle(title, fontsize=16, fontweight="bold", color=C_LABEL, y=1.02)

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {filename}")


save_figure(generator_layers,     "Generador-Conv",     "conv_generator.png",     figw=24)
save_figure(discriminator_layers, "Discriminador-Conv", "conv_discriminator.png", figw=24)