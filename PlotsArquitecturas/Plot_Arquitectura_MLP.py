import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# Paleta 
C_LINEAR   = "#378ADD"   # azul -> Linear
C_RELU     = "#1D9E75"   # verde -> ReLU
C_LRELU    = "#BA7517"   # ámbar -> LeakyReLU
C_TANH     = "#7F77DD"   # púrpura -> Tanh
C_SIGMOID  = "#D85A30"   # coral -> Sigmoid
C_TEXT     = "#FFFFFF"
C_LABEL    = "#333333"
C_ARROW    = "#888888"
BG         = "#F8F8F6"

# Arquitecturas
CHANNELS = 1          # 1 para MNIST/FashionMNIST, 3 para SVHN
IMG_SIZE = 32
IMG_DIM  = CHANNELS * IMG_SIZE * IMG_SIZE
Z_DIM    = 100

discriminator_layers = [
    {"label": "Input\nIMG_DIM",             "type": "input",  "color": "#888780", "units": IMG_DIM, "units_label": "IMG_DIM"},
    {"label": "Linear\nIMG_DIM→512",        "type": "linear", "color": C_LINEAR,  "units": 512},
    {"label": "LeakyReLU\n0.2",             "type": "act",    "color": C_LRELU},
    {"label": "Linear\n512→256",            "type": "linear", "color": C_LINEAR,  "units": 256},
    {"label": "LeakyReLU\n0.2",             "type": "act",    "color": C_LRELU},
    {"label": "Linear\n256→1",              "type": "linear", "color": C_LINEAR,  "units": 1},
    # Sin Sigmoid para crítico WGAN, salida escalar sin acotar y con para Vanilla
    {"label": "Sigmoid",                    "type": "act",    "color": C_SIGMOID},
]

generator_layers = [
    {"label": f"Noise\n{Z_DIM}",            "type": "input",  "color": "#888780", "units": Z_DIM},
    {"label": "Linear\n100→256",            "type": "linear", "color": C_LINEAR,  "units": 256},
    {"label": "ReLU",                       "type": "act",    "color": C_RELU},
    {"label": "Linear\n256→512",            "type": "linear", "color": C_LINEAR,  "units": 512},
    {"label": "ReLU",                       "type": "act",    "color": C_RELU},
    {"label": "Linear\n512→IMG_DIM",        "type": "linear", "color": C_LINEAR,  "units": IMG_DIM, "units_label": "IMG_DIM"},
    {"label": "Tanh",                       "type": "act",    "color": C_TANH},
]


def units_to_height(units, min_h=0.5, max_h=4.0, ref=IMG_DIM):
    """Altura proporcional al log del número de unidades."""
    if units is None:
        return min_h
    return min_h + (max_h - min_h) * (np.log1p(units) / np.log1p(ref))

def draw_network(ax, layers, title, start_x=0.4, block_w=1.0, gap=0.85):
    """Dibuja una red como bloques isométricos apilados en horizontal."""
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    iso_dx = 0.18
    iso_dy = 0.12       

    xs = []
    x = start_x
    
    last_units = 100 # Valor inicial por defecto
    
    for i, layer in enumerate(layers):
        # Heredar unidades si no están definidas
        current_units = layer.get("units", last_units)
        last_units = current_units # Actualizar para la siguiente capa
        
        h = units_to_height(current_units)
        y = (5.0 - h) / 2          # centrado vertical
        c = layer["color"]

        # cara frontal
        front = mpatches.FancyBboxPatch(
            (x, y), block_w, h,
            boxstyle="round,pad=0.06",
            fc=c, ec="white", lw=1.2, zorder=3
        )
        ax.add_patch(front)

        # cara superior
        top_x = [x, x + block_w, x + block_w + iso_dx, x + iso_dx, x]
        top_y = [y + h, y + h, y + h + iso_dy, y + h + iso_dy, y + h]
        ax.fill(top_x, top_y, color=c, alpha=0.55, zorder=2)
        ax.plot(top_x, top_y, color="white", lw=0.8, zorder=4)

        # cara lateral derecha
        side_x = [x + block_w, x + block_w + iso_dx, x + block_w + iso_dx, x + block_w, x + block_w]
        side_y = [y, y + iso_dy, y + h + iso_dy, y + h, y]
        ax.fill(side_x, side_y, color=c, alpha=0.30, zorder=2)
        ax.plot(side_x, side_y, color="white", lw=0.8, zorder=4)

        # Sistema de etiquetas
        fontsize = 7.5 if "\n" in layer["label"] else 8.0
        text_x = x + block_w / 2 + iso_dx / 2

        if layer["type"] in ["input", "linear"]:
            # Carril inferior para Input y Linear
            label_y = -0.2
            ax.text(
                text_x, label_y, layer["label"],
                ha="center", va="top",
                fontsize=fontsize, color=c, fontweight="bold",
                linespacing=1.35, zorder=5
            )
            # Línea punteada conector inferior
            ax.plot(
                [text_x, text_x],
                [y - 0.05, label_y + 0.1],
                color=c, lw=1.2, ls=":", alpha=0.5, zorder=4
            )
        elif layer["type"] == "act":
            # Carril superior para activaciones
            label_y = 5.3
            ax.text(
                text_x, label_y, layer["label"],
                ha="center", va="center",
                fontsize=7.5, color=c, fontweight="bold",
                linespacing=1.3, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=c, lw=1.0, alpha=0.95)
            )
            # Línea punteada conector superior
            ax.plot(
                [text_x, text_x],
                [y + h + iso_dy + 0.05, label_y - 0.25],
                color=c, lw=1.2, ls=":", alpha=0.5, zorder=4
            )

        xs.append((x, block_w, y, h))

        # flecha de conexión
        if i < len(layers) - 1:
            # Empezamos la flecha después del efecto 3D lateral
            start_arrow = x + block_w + iso_dx + 0.05
            # Terminamos la flecha un poco antes de la siguiente caja
            end_arrow = x + block_w + gap - 0.2
            
            ax.annotate(
                "", xy=(end_arrow, 2.5),
                xytext=(start_arrow, 2.5),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.5),
                zorder=4
            )

        x += block_w + gap

    # Ajuste de límites para que quepan los carriles superior e inferior
    ax.set_xlim(-0.2, x + 0.2)
    ax.set_ylim(-1.2, 6.0)
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=C_LABEL, pad=10, loc="left")


NOTE = f"IMG_DIM = CHANNELS × IMG_SIZE × IMG_SIZE"

# Generador
fig, ax_net = plt.subplots(1, 1, figsize=(16, 5.0), facecolor=BG)

fig.text(0.5, 0.90, NOTE, ha="center", va="top",
         fontsize=9, color="#666666", style="italic")

draw_network(ax_net, generator_layers, "")
plt.savefig("mlp_generator.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("Guardado: mlp_generator.png")

# Discriminador
fig, ax_net = plt.subplots(1, 1, figsize=(16, 5.0), facecolor=BG)

fig.text(0.5, 0.90, NOTE, ha="center", va="top",
         fontsize=9, color="#666666", style="italic")

draw_network(ax_net, discriminator_layers, "")
plt.savefig("mlp_discriminator.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("Guardado: mlp_discriminator.png")