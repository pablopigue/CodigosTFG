"""
Plots de las arquitecturas de CycleGAN.

Genera tres figuras:
    A. cyclegan_generator.png    — Generador ResNet-9 con detalle de un
                                   bloque residual.
    B. cyclegan_discriminator.png — Discriminador PatchGAN 70×70.
    C. cyclegan_cycle.png         — Esquema conceptual del ciclo completo.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# Paleta común con los plots de DCGAN
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

C_RESBLOCK   = "#9B59B6"
C_REFLECT    = "#16A085"
C_INSTNORM   = "#1D9E75"
C_PATCH      = "#E67E22"
C_GEN_AB     = "#378ADD"
C_GEN_BA     = "#D85A30"
C_DISC_A     = "#BA7517"
C_DISC_B     = "#16A085"
C_DOMAIN_A   = "#E74C3C"
C_DOMAIN_B   = "#F39C12"


# ============================================================================
# UTILIDADES
# ============================================================================

def spatial_to_w(s, min_w=0.45, max_w=0.85, ref=256):
    return min_w + (max_w - min_w) * (np.log1p(s) / np.log1p(ref))


def channels_to_h(c, min_h=0.5, max_h=3.4, ref=512):
    return min_h + (max_h - min_h) * (np.log1p(c) / np.log1p(ref))


def is_activation_layer(layer):
    return "spatial" not in layer and "channels" not in layer


def draw_volume(ax, x, y_center, w, h, color, label, label_position="main",
                zorder=3, narrow=False):
    """
    Dibuja un bloque 3D y su etiqueta alineada en carriles.
    Soporta carriles de activación escalonados (activation_1, activation_2).
    """
    if narrow:
        iso_dx = w * 0.30
        iso_dy = w * 0.20
    else:
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
    tx = [x, x + w, x + w + iso_dx, x + iso_dx, x]
    ty = [y + h, y + h, y + h + iso_dy, y + h + iso_dy, y + h]
    ax.fill(tx, ty, color=color, alpha=0.45, zorder=zorder - 1)
    ax.plot(tx, ty, color="white", lw=0.7, zorder=zorder + 1)

    # Right face
    sx = [x + w, x + w + iso_dx, x + w + iso_dx, x + w, x + w]
    sy = [y, y + iso_dy, y + h + iso_dy, y + h, y]
    ax.fill(sx, sy, color=color, alpha=0.25, zorder=zorder - 1)
    ax.plot(sx, sy, color="white", lw=0.7, zorder=zorder + 1)

    # Label rendering (Carriles fijos)
    fontsize = 6.0 if "\n" in label else 6.8
    text_x = x + w / 2 + iso_dx / 2

    if label_position == "main":
        # Carril inferior para las capas físicas (Conv, Input, etc)
        label_y = -0.1
        ax.text(
            text_x, label_y, label,
            ha="center", va="top",
            fontsize=fontsize, color=color, fontweight="bold",
            linespacing=1.35, zorder=zorder + 2
        )
        ax.plot(
            [text_x, text_x],
            [y - 0.05, label_y + 0.1],
            color=color, lw=1.2, ls=":", alpha=0.5, zorder=zorder + 1
        )
    elif label_position.startswith("activation"):
        # Carril superior escalonado para evitar que InstNorm y ReLU choquen
        level = 1
        if "_" in label_position:
            level = int(label_position.split("_")[1])
            
        # Nivel 1 se queda en 5.9, Nivel 2 sube a 6.8
        label_y = 5.0 + (level * 0.9) 
        
        ax.text(
            text_x, label_y, label,
            ha="center", va="center",
            fontsize=6.2, color=color, fontweight="bold",
            linespacing=1.3, zorder=zorder + 2,
            bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=color, lw=1.0, alpha=0.95)
        )
        ax.plot(
            [text_x, text_x],
            [y + h + iso_dy + 0.05, label_y - 0.25],
            color=color, lw=1.2, ls=":", alpha=0.5, zorder=zorder + 1
        )

    return x + w + iso_dx


def draw_arrow(ax, x_from, x_to, y, color=C_ARROW, lw=1.1, style="->"):
    ax.annotate(
        "", xy=(x_to, y), xytext=(x_from, y),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=2
    )


# ============================================================================
# A. GENERADOR ResNet-9
# ============================================================================

# Encoder: reduce 256×256×3 -> 64×64×256
encoder_layers = [
    dict(spatial=256, channels=3,   label="Input\n256×256\n×3",    color=C_INPUT),
    dict(spatial=256, channels=64,  label="RPad+Conv 7×7\n→256×256×64",
         color=C_CONV),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="ReLU", color=C_RELU, narrow=True),
    dict(spatial=128, channels=128, label="Conv 3×3 s=2\n→128×128×128",
         color=C_CONV),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="ReLU", color=C_RELU, narrow=True),
    dict(spatial=64,  channels=256, label="Conv 3×3 s=2\n→64×64×256",
         color=C_CONV),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="ReLU", color=C_RELU, narrow=True),
    dict(spatial=64,  channels=256, label="Output\n64×64\n×256",   color=C_INPUT),
]

# Bottleneck: 9 × ResidualBlock (64×64×256)
bottleneck_layers = [
    dict(spatial=64, channels=256, label="Input\n64×64\n×256",     color=C_INPUT),
    dict(spatial=64, channels=256, label="ResidualBlock\n× 9\n(64×64×256)",
         color=C_RESBLOCK),
    dict(spatial=64, channels=256, label="Output\n64×64\n×256",    color=C_INPUT),
]

# Decoder: 64×64×256 -> 256×256×3
decoder_layers = [
    dict(spatial=64,  channels=256, label="Input\n64×64\n×256",    color=C_INPUT),
    dict(spatial=128, channels=128, label="ConvT 3×3 s=2\n→128×128×128",
         color=C_CONVT),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="ReLU", color=C_RELU, narrow=True),
    dict(spatial=256, channels=64,  label="ConvT 3×3 s=2\n→256×256×64",
         color=C_CONVT),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="ReLU", color=C_RELU, narrow=True),
    dict(spatial=256, channels=3,   label="RPad+Conv 7×7\n→256×256×3",
         color=C_CONV),
    dict(label="Tanh", color=C_TANH, narrow=True),
    dict(spatial=256, channels=3,   label="Output\n256×256\n×3",   color=C_INPUT),
]


def draw_residual_block_expanded(ax, x_start, y_center, color=C_RESBLOCK):
    box_w = 0.40
    box_h = 0.62
    gap = 0.20

    components = [
        ("ReflectionPad", C_REFLECT),
        ("Conv 3×3", C_CONV),
        ("InstanceNorm", C_INSTNORM),
        ("ReLU", C_RELU),
        ("ReflectionPad", C_REFLECT),
        ("Conv 3×3", C_CONV),
        ("InstanceNorm", C_INSTNORM),
    ]

    x = x_start
    for i, (label, col) in enumerate(components):
        rect = FancyBboxPatch(
            (x, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.04",
            fc=col, ec="white", lw=0.8, zorder=3
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2, y_center, label,
            ha="center", va="center", fontsize=5.5, color="white",
            fontweight="bold", zorder=4,
            path_effects=[pe.withStroke(linewidth=0.8, foreground="black")]
        )
        
        if i < len(components) - 1:
            draw_arrow(ax, x + box_w + 0.02, x + box_w + gap - 0.02,
                       y_center, color=C_ARROW, lw=0.9)
        x += box_w + gap

    # Sumador
    sum_x = x  
    sum_box = FancyBboxPatch(
        (sum_x, y_center - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.04",
        fc="white", ec=color, lw=2.0, zorder=4
    )
    ax.add_patch(sum_box)
    ax.text(
        sum_x + box_w / 2, y_center, "+",
        ha="center", va="center", fontsize=20, color=color,
        fontweight="bold", zorder=5
    )
    draw_arrow(ax, sum_x - gap + 0.02, sum_x - 0.02, y_center,
               color=C_ARROW, lw=0.9)

    # Skip connection
    skip_y_top = y_center + box_h / 2 + 0.55
    ax.plot(
        [x_start - 0.10, x_start - 0.10],
        [y_center, skip_y_top],
        color=color, lw=1.5, zorder=2
    )
    ax.plot(
        [x_start - 0.10, sum_x + box_w / 2],
        [skip_y_top, skip_y_top],
        color=color, lw=1.5, zorder=2
    )
    ax.annotate(
        "", xy=(sum_x + box_w / 2, y_center + box_h / 2 + 0.02),
        xytext=(sum_x + box_w / 2, skip_y_top),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        zorder=2
    )
    ax.text(
        x_start + 0.50, skip_y_top + 0.18, "skip connection",
        ha="left", va="bottom", fontsize=8, color=color,
        fontweight="bold", style="italic"
    )

    ax.text(
        x_start - 0.10, y_center - box_h / 2 - 0.22, "in",
        ha="center", va="top", fontsize=7, color=C_LABEL, style="italic"
    )
    ax.text(
        sum_x + box_w / 2, y_center - box_h / 2 - 0.22, "out",
        ha="center", va="top", fontsize=7, color=C_LABEL, style="italic"
    )

    draw_arrow(ax, x_start - 0.07, x_start - 0.02, y_center,
               color=C_ARROW, lw=0.9)

    return sum_x + box_w + 0.1


def draw_subnetwork(layers, title, filename, figw=14):
    fig, ax_main = plt.subplots(figsize=(figw, 5.5), facecolor=BG)

    ax_main.set_facecolor(BG)
    ax_main.set_aspect("equal")
    ax_main.axis("off")

    CENTER_Y = 2.8
    x = 0.3
    last_spatial = layers[0].get("spatial", 256)
    last_channels = layers[0].get("channels", 3)

    # Asignación escalonada
    label_positions = []
    act_level = 1
    for layer in layers:
        if is_activation_layer(layer):
            label_positions.append(f"activation_{act_level}")
            act_level = 2 if act_level == 1 else 1 # Alterna nivel
        else:
            label_positions.append("main")
            act_level = 1 # Tras una capa principal, reinicia el nivel superior

    for i, layer in enumerate(layers):
        is_act = is_activation_layer(layer)
        narrow = layer.get("narrow", False)

        if is_act:
            current_spatial = last_spatial
            current_channels = last_channels
            w = 0.32
            h = 1.4
        else:
            current_spatial = layer.get("spatial", last_spatial)
            current_channels = layer.get("channels", last_channels)
            last_spatial = current_spatial
            last_channels = current_channels
            w = spatial_to_w(current_spatial)
            h = channels_to_h(current_channels)

        right_edge = draw_volume(
            ax_main, x, CENTER_Y, w, h,
            layer["color"], layer["label"],
            label_position=label_positions[i],
            narrow=narrow
        )

        if i < len(layers) - 1:
            next_layer = layers[i + 1]
            next_is_act = is_activation_layer(next_layer)
            if (is_act and next_is_act) or (not is_act and next_is_act):
                gap = 0.28
            else:
                gap = 0.55

            arr_x0 = right_edge + 0.05
            arr_x1 = right_edge + gap - 0.05
            draw_arrow(ax_main, arr_x0, arr_x1, CENTER_Y, lw=0.9)
            x = right_edge + gap
        else:
            x = right_edge

    ax_main.set_xlim(-0.1, x + 0.3)
    ax_main.set_ylim(-1.5, 7.8) # Ampliado para el carril doble

    fig.suptitle(
        title,
        fontsize=15, fontweight="bold", color=C_LABEL, y=1.00
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {filename}")


def draw_residual_block_detail(filename, figw=14):
    fig, ax = plt.subplots(figsize=(figw, 4.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_residual_block_expanded(ax, x_start=0.4, y_center=1.3)

    ax.set_xlim(0, 5.4)
    ax.set_ylim(0.0, 2.7)

    fig.suptitle(
        "Estructura interna de un ResidualBlock",
        fontsize=15, fontweight="bold", color=C_RESBLOCK, y=1.00
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {filename}")


# ============================================================================
# B. DISCRIMINADOR PatchGAN 70×70
# ============================================================================

discriminator_layers = [
    dict(spatial=256, channels=3,   label="Input\n256×256\n×3",
         color=C_INPUT),
    dict(spatial=128, channels=64,  label="Conv 4×4 s=2\n→128×128×64",
         color=C_CONV),
    dict(label="LeakyReLU", color=C_LRELU, narrow=True),
    dict(spatial=64,  channels=128, label="Conv 4×4 s=2\n→64×64×128",
         color=C_CONV),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="LeakyReLU", color=C_LRELU, narrow=True),
    dict(spatial=32,  channels=256, label="Conv 4×4 s=2\n→32×32×256",
         color=C_CONV),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="LeakyReLU", color=C_LRELU, narrow=True),
    dict(spatial=31,  channels=512, label="Conv 4×4 s=1\n→31×31×512",
         color=C_CONV),
    dict(label="InstNorm", color=C_INSTNORM, narrow=True),
    dict(label="LeakyReLU", color=C_LRELU, narrow=True),
    dict(spatial=30,  channels=1,   label="Conv 4×4 s=1\n→30×30×1",
         color=C_PATCH),
]


def draw_simple_network(layers, title, filename, figw=22, subtitle=None):
    fig, ax = plt.subplots(figsize=(figw, 5.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    CENTER_Y = 2.8
    x = 0.3
    last_spatial = 256
    last_channels = 3

    # Asignación escalonada
    label_positions = []
    act_level = 1
    for layer in layers:
        if is_activation_layer(layer):
            label_positions.append(f"activation_{act_level}")
            act_level = 2 if act_level == 1 else 1 # Alterna nivel
        else:
            label_positions.append("main")
            act_level = 1 # Tras una capa principal, reinicia el nivel superior

    for i, layer in enumerate(layers):
        is_act = is_activation_layer(layer)
        narrow = layer.get("narrow", False)

        if is_act:
            w = 0.32
            h = 1.4
        else:
            current_spatial = layer.get("spatial", last_spatial)
            current_channels = layer.get("channels", last_channels)
            last_spatial = current_spatial
            last_channels = current_channels
            w = spatial_to_w(current_spatial)
            h = channels_to_h(current_channels)

        right_edge = draw_volume(
            ax, x, CENTER_Y, w, h,
            layer["color"], layer["label"],
            label_position=label_positions[i],
            narrow=narrow
        )

        if i < len(layers) - 1:
            next_layer = layers[i + 1]
            next_is_act = is_activation_layer(next_layer)
            if (is_act and next_is_act) or (not is_act and next_is_act):
                gap = 0.28
            else:
                gap = 0.55

            arr_x0 = right_edge + 0.05
            arr_x1 = right_edge + gap - 0.05
            draw_arrow(ax, arr_x0, arr_x1, CENTER_Y, lw=0.9)
            x = right_edge + gap
        else:
            x = right_edge

    ax.set_xlim(-0.1, x + 0.3)
    ax.set_ylim(-1.5, 8.0) # Ampliado para el carril doble

    fig.suptitle(title, fontsize=16, fontweight="bold",
                 color=C_LABEL, y=1.02)
    if subtitle:
        ax.text(
            (x + 0.3) / 2, 7.6, subtitle, # Subtítulo subido también
            ha="center", va="center", fontsize=10, color=C_LABEL,
            style="italic"
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {filename}")


# ============================================================================
# C. ESQUEMA DEL CICLO COMPLETO
# ============================================================================

def draw_cycle_diagram(filename):
    fig, ax = plt.subplots(figsize=(14, 8.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)

    yA = 8.2
    yB = 2.8

    real_A_x = 1.5
    fake_B_x = 6.8
    recov_A_x = 12.5

    real_B_x = 1.5
    fake_A_x = 6.8
    recov_B_x = 12.5

    box_w = 1.9
    box_h = 1.4

    def img_box(ax, cx, cy, label_top, label_formula, color, dashed=False):
        ls = "--" if dashed else "-"
        rect = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08",
            fc="white", ec=color, lw=2.4, ls=ls, zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.30, label_top, ha="center", va="center",
                fontsize=11, color=color, fontweight="bold", zorder=4)
        if label_formula:
            ax.text(cx, cy - 0.28, label_formula, ha="center", va="center",
                    fontsize=8.5, color=color, style="italic", zorder=4)

    img_box(ax, real_A_x, yA, "Real A", "(manzana)", C_DOMAIN_A)
    img_box(ax, fake_B_x, yA, "Fake B", r"$G_{AB}(A)$", C_DOMAIN_B)
    img_box(ax, recov_A_x, yA, "Rec A", r"$G_{BA}(G_{AB}(A))$", C_DOMAIN_A,
            dashed=True)

    img_box(ax, real_B_x, yB, "Real B", "(naranja)", C_DOMAIN_B)
    img_box(ax, fake_A_x, yB, "Fake A", r"$G_{BA}(B)$", C_DOMAIN_A)
    img_box(ax, recov_B_x, yB, "Rec B", r"$G_{AB}(G_{BA}(B))$", C_DOMAIN_B,
            dashed=True)

    def gen_box(ax, cx, cy, label, color):
        rect = FancyBboxPatch(
            (cx - 0.50, cy - 0.40), 1.0, 0.80,
            boxstyle="round,pad=0.05",
            fc=color, ec="white", lw=1.6, zorder=4
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold", zorder=5,
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    gen_box(ax, (real_A_x + fake_B_x) / 2, yA, r"$G_{AB}$", C_GEN_AB)
    gen_box(ax, (fake_A_x + recov_B_x) / 2, yB, r"$G_{AB}$", C_GEN_AB)
    gen_box(ax, (fake_B_x + recov_A_x) / 2, yA, r"$G_{BA}$", C_GEN_BA)
    gen_box(ax, (real_B_x + fake_A_x) / 2, yB, r"$G_{BA}$", C_GEN_BA)

    def cycle_arrow(ax, x_from, x_to, y, color):
        gen_left = (x_from + x_to) / 2 - 0.55
        gen_right = (x_from + x_to) / 2 + 0.55
        ax.annotate("", xy=(gen_left, y), xytext=(x_from + box_w / 2, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2),
                    zorder=2)
        ax.annotate("", xy=(x_to - box_w / 2, y), xytext=(gen_right, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2),
                    zorder=2)

    cycle_arrow(ax, real_A_x, fake_B_x, yA, C_GEN_AB)
    cycle_arrow(ax, fake_B_x, recov_A_x, yA, C_GEN_BA)
    cycle_arrow(ax, real_B_x, fake_A_x, yB, C_GEN_BA)
    cycle_arrow(ax, fake_A_x, recov_B_x, yB, C_GEN_AB)

    def disc_box(ax, cx, cy, label, color):
        rect = FancyBboxPatch(
            (cx - 0.50, cy - 0.40), 1.0, 0.80,
            boxstyle="round,pad=0.05",
            fc=color, ec="white", lw=1.6, zorder=4
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold", zorder=5,
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    DB_x, DB_y = 5.5, 5.5
    DA_x, DA_y = 8.1, 5.5

    disc_box(ax, DB_x, DB_y, r"$D_B$", C_DISC_B)
    disc_box(ax, DA_x, DA_y, r"$D_A$", C_DISC_A)

    def disc_arrow(ax, x_src, y_src, x_dst, y_dst, color):
        ax.annotate(
            "", xy=(x_dst, y_dst), xytext=(x_src, y_src),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4,
                            ls="--", alpha=0.85, shrinkA=2, shrinkB=4),
            zorder=2
        )

    disc_arrow(ax, fake_B_x - 0.4, yA - box_h / 2, DB_x + 0.2, DB_y + 0.4,
               C_DISC_B)
    disc_arrow(ax, real_B_x + box_w / 2 - 0.2, yB + box_h / 2,
               DB_x - 0.4, DB_y - 0.4, C_DISC_B)

    disc_arrow(ax, real_A_x + box_w / 2 - 0.2, yA - box_h / 2,
               DA_x - 0.4, DA_y + 0.4, C_DISC_A)
    disc_arrow(ax, fake_A_x + 0.4, yB + box_h / 2,
               DA_x + 0.2, DA_y - 0.4, C_DISC_A)

    ax.text(DB_x - 0.5, DB_y + 0.8, r"$\mathcal{L}_{\rm GAN}$",
            fontsize=10, color=C_DISC_B, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=C_DISC_B,
                      lw=0.8, alpha=0.9))
    ax.text(DA_x + 0.5, DA_y + 0.8, r"$\mathcal{L}_{\rm GAN}$",
            fontsize=10, color=C_DISC_A, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=C_DISC_A,
                      lw=0.8, alpha=0.9))

    def cycle_loss_arrow(ax, x_real, x_rec, y, label, color, top=True):
        if top:
            connection = "arc3,rad=0.35"
            arc_y = y + 1.5
            y_end = y + box_h / 2
        else:
            connection = "arc3,rad=-0.35"
            arc_y = y - 1.5
            y_end = y - box_h / 2

        arrow = FancyArrowPatch(
            (x_real, y_end), (x_rec, y_end),
            connectionstyle=connection,
            arrowstyle="<->", color=color, lw=1.6, ls=":", alpha=0.95,
            zorder=2
        )
        ax.add_patch(arrow)
        ax.text((x_real + x_rec) / 2, arc_y, label,
                ha="center", va="center",
                fontsize=10, color=color, fontweight="bold", style="italic",
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec=color, lw=1.1, alpha=0.97))

    cycle_loss_arrow(ax, real_A_x, recov_A_x, yA,
                     r"$\mathcal{L}_{\rm cycle}\;(A \to B \to A)$",
                     C_DOMAIN_A, top=True)
    cycle_loss_arrow(ax, real_B_x, recov_B_x, yB,
                     r"$\mathcal{L}_{\rm cycle}\;(B \to A \to B)$",
                     C_DOMAIN_B, top=False)

    formula = (r"$\mathcal{L}_{\rm total} = "
               r"\mathcal{L}_{\rm GAN}(G_{AB}, D_B) + "
               r"\mathcal{L}_{\rm GAN}(G_{BA}, D_A) + "
               r"\lambda_{\rm cycle}\,\mathcal{L}_{\rm cycle} + "
               r"\lambda_{\rm id}\,\mathcal{L}_{\rm identity}$")
    ax.text(8, 0.4, formula, ha="center", va="center",
            fontsize=12.5, color=C_LABEL,
            bbox=dict(boxstyle="round,pad=0.4", fc=BG,
                      ec=C_LABEL, lw=1, alpha=0.9))

    legend_x = 14.5
    legend_y0 = 9
    legend_items = [
        (C_GEN_AB, r"$G_{AB}$: A → B"),
        (C_GEN_BA, r"$G_{BA}$: B → A"),
        (C_DISC_A, r"$D_A$: discrimina A"),
        (C_DISC_B, r"$D_B$: discrimina B"),
    ]
    ax.text(legend_x, legend_y0 + 0.10, "Componentes",
            ha="left", va="bottom", fontsize=10, color=C_LABEL,
            fontweight="bold")
    for i, (col, txt) in enumerate(legend_items):
        y_item = legend_y0 - 0.30 - i * 0.40
        rect = FancyBboxPatch(
            (legend_x, y_item - 0.13), 0.30, 0.26,
            boxstyle="round,pad=0.02", fc=col, ec="white", lw=0.6
        )
        ax.add_patch(rect)
        ax.text(legend_x + 0.40, y_item, txt, ha="left", va="center",
                fontsize=9, color=C_LABEL)

    ax.text(real_A_x - 0.2, yA + box_h / 2 + 0.45,
            "Ciclo A → B → A",
            fontsize=11, color=C_DOMAIN_A, fontweight="bold",
            ha="left", va="bottom", style="italic")
    ax.text(real_B_x - 0.2, yB - box_h / 2 - 0.50,
            "Ciclo B → A → B",
            fontsize=11, color=C_DOMAIN_B, fontweight="bold",
            ha="left", va="top", style="italic")

    fig.suptitle("Esquema del entrenamiento de CycleGAN",
                 fontsize=15, fontweight="bold", color=C_LABEL, y=0.85)

    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {filename}")


def draw_generator_overview(filename, figw=12):
    fig, ax = plt.subplots(figsize=(figw, 2.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    blocks = [
        dict(label="Encoder",    sublabel="3 × Conv↓",
             color=C_CONV),
        dict(label="Bottleneck", sublabel="9 × ResidualBlock",
             color=C_RESBLOCK),
        dict(label="Decoder",    sublabel="2 × ConvT↑ + Conv 7×7",
             color=C_CONVT),
    ]

    flow_dims = ["256×256×3", "64×64×256", "64×64×256", "256×256×3"]

    box_w = 2.6
    box_h = 1.4
    gap = 1.40
    y_center = 1.5

    pre_margin = 1.30
    x = pre_margin

    block_centers = []
    for i, blk in enumerate(blocks):
        rect = FancyBboxPatch(
            (x, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.10",
            fc=blk["color"], ec="white", lw=1.5, zorder=3
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2, y_center + 0.22, blk["label"],
            ha="center", va="center", fontsize=13, color="white",
            fontweight="bold", zorder=4,
            path_effects=[pe.withStroke(linewidth=1.0, foreground="black")]
        )
        ax.text(
            x + box_w / 2, y_center - 0.25, blk["sublabel"],
            ha="center", va="center", fontsize=9.5, color="white",
            style="italic", zorder=4,
            path_effects=[pe.withStroke(linewidth=0.7, foreground="black")]
        )
        block_centers.append(x + box_w / 2)

        if i < len(blocks) - 1:
            ax.annotate(
                "", xy=(x + box_w + gap - 0.05, y_center),
                xytext=(x + box_w + 0.05, y_center),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=2.0),
                zorder=2
            )
        x += box_w + gap

    x_end = x - gap + box_w  

    flow_x = [
        pre_margin - 0.65,                                 
        block_centers[0] + box_w / 2 + gap / 2,            
        block_centers[1] + box_w / 2 + gap / 2,            
        x_end + 0.65,                                      
    ]

    ax.annotate(
        "", xy=(pre_margin - 0.05, y_center),
        xytext=(flow_x[0] + 0.30, y_center),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=2.0),
        zorder=2
    )
    ax.annotate(
        "", xy=(flow_x[3] - 0.30, y_center),
        xytext=(x_end + 0.05, y_center),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=2.0),
        zorder=2
    )

    for i, (fx, dim) in enumerate(zip(flow_x, flow_dims)):
        if i == 0 or i == 3:
            y_label = y_center
            va = "center"
        else:
            y_label = y_center + 0.45
            va = "bottom"
        ax.text(
            fx, y_label, dim,
            ha="center", va=va, fontsize=8.5, color=C_LABEL,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec=C_LABEL,
                      lw=0.6, alpha=0.95)
        )

    ax.set_xlim(0, x_end + 1.4)
    ax.set_ylim(-0.3, 3.0)

    fig.suptitle(
        "Generador CycleGAN — vista de alto nivel",
        fontsize=13, fontweight="bold", color=C_LABEL, y=1.02
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {filename}")


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    draw_generator_overview("cyclegan_generator_overview.png", figw=12)
    draw_subnetwork(
        encoder_layers,
        "Encoder del generador CycleGAN — 256×256×3 → 64×64×256",
        "cyclegan_generator_encoder.png",
        figw=14
    )
    draw_subnetwork(
        bottleneck_layers,
        "Bottleneck del generador CycleGAN — 9 × ResidualBlock (64×64×256)",
        "cyclegan_generator_bottleneck.png",
        figw=8
    )
    draw_subnetwork(
        decoder_layers,
        "Decoder del generador CycleGAN — 64×64×256 → 256×256×3",
        "cyclegan_generator_decoder.png",
        figw=14
    )
    draw_residual_block_detail("cyclegan_residual_block.png", figw=14)
    draw_simple_network(
        discriminator_layers,
        "Discriminador CycleGAN — PatchGAN 70×70",
        "cyclegan_discriminator.png",
        figw=24,
        subtitle=("Salida: mapa 30×30 de decisiones por parche, "
                  "campo receptivo 70×70 píxeles. Sin sigmoide final por usar LSGAN.")
    )
    draw_cycle_diagram("cyclegan_cycle.png")
    print("\n¡Hecho! Siete figuras generadas.")