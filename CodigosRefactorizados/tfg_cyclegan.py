"""
Entrenamiento de CycleGAN sobre apple2orange.
"""

import os
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torchmetrics.image import FrechetInceptionDistance
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================
plt.switch_backend('agg')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros (Zhu et al. 2017)
EPOCHS = 200
DECAY_START_EPOCH = 100      # Comienzo del decay lineal del learning rate
BATCH_SIZE = 1
LR = 0.0002
BETA1 = 0.5
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 5.0

# Frecuencias
SAVE_IMG_FREQ = 5
FID_FREQ = 10                # Calcular FID cada N épocas
FID_BATCH_SIZE = 8           # Tamaño del batch al generar imágenes para FID

# Rutas
EXPERIMENT_DIR = "/mnt/homeGPU/pablomarpa/CodigosTFG/tfg_cyclegan_apple2orange"
DATASET_DIR = "./apple2orange"

# Crear estructura de carpetas
for sub in ['images', 'models', 'plots', 'logs']:
    os.makedirs(f"{EXPERIMENT_DIR}/{sub}", exist_ok=True)

print(f"Iniciando entrenamiento CycleGAN en: {DEVICE}", flush=True)
print(f"Épocas: {EPOCHS} (decay lineal a partir de la {DECAY_START_EPOCH+1})",
      flush=True)


# ============================================================
# 1. DATASET
# ============================================================
class ImageDataset(Dataset):
    """Dataset emparejado aleatoriamente para CycleGAN."""

    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms_
        path_A = os.path.join(root, f'{mode}A')
        path_B = os.path.join(root, f'{mode}B')

        self.files_A = sorted([
            os.path.join(path_A, x) for x in os.listdir(path_A)
            if x.endswith(('.jpg', '.png'))
        ])
        self.files_B = sorted([
            os.path.join(path_B, x) for x in os.listdir(path_B)
            if x.endswith(('.jpg', '.png'))
        ])

    def __getitem__(self, index):
        item_A = self.transform(
            Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        item_B = self.transform(
            Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            .convert('RGB'))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


# Transformaciones
train_transforms = transforms.Compose([
    transforms.Resize((286, 286), Image.BICUBIC),     
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Sin augmentation para test
test_transforms = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Cargando datasets...", flush=True)
train_dataset = ImageDataset(DATASET_DIR, transforms_=train_transforms,
                             mode='train')
test_dataset = ImageDataset(DATASET_DIR, transforms_=test_transforms,
                            mode='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)

# Loader para FID con un batch size mayor
test_loader_fid = DataLoader(test_dataset, batch_size=FID_BATCH_SIZE,
                             shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)} pares | Test: {len(test_dataset)} pares",
      flush=True)


# ============================================================
# 2. MODELOS
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9):
        super().__init__()
        channels = input_shape[0]

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(channels, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Bloques residuales
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Salida
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, channels, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN 70x70."""

    def __init__(self, input_shape):
        super().__init__()
        channels, _, _ = input_shape

        def block(in_filters, out_filters, normalize=True, stride=2):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Capas con stride=2
            *block(channels, 64, normalize=False, stride=2),
            *block(64, 128, stride=2),
            *block(128, 256, stride=2),
            
            # Capa de 512 canales con stride=1
            *block(256, 512, stride=1),
            
            # Capa de salida con stride=1
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def forward(self, img):
        return self.model(img)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif (classname.find('BatchNorm2d') != -1
          or classname.find('InstanceNorm2d') != -1):
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


# ============================================================
# 3. PREPARACIÓN
# ============================================================
input_shape = (3, 256, 256)

G_AB = GeneratorResNet(input_shape, num_residual_blocks=9).to(DEVICE)
G_BA = GeneratorResNet(input_shape, num_residual_blocks=9).to(DEVICE)
D_A = Discriminator(input_shape).to(DEVICE)
D_B = Discriminator(input_shape).to(DEVICE)

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G = optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr=LR, betas=(BETA1, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(BETA1, 0.999))


# Decay lineal: LR constante hasta DECAY_START_EPOCH, luego baja a 0
def lambda_rule(epoch):
    """LR multiplier: 1.0 hasta DECAY_START_EPOCH, decay lineal a 0 después."""
    if epoch < DECAY_START_EPOCH:
        return 1.0
    return max(0.0, 1.0 - (epoch - DECAY_START_EPOCH)
               / float(EPOCHS - DECAY_START_EPOCH))


lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G,
                                             lr_lambda=lambda_rule)
lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                               lr_lambda=lambda_rule)
lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                               lr_lambda=lambda_rule)


class ReplayBuffer:
    """Buffer histórico de imágenes generadas."""

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


# ============================================================
# 4. CÁLCULO DE FID
# ============================================================
def tensor_to_uint8(t):
    """Convierte tensor en [-1, 1] a uint8 [0, 255] con clamp de seguridad."""
    return (((t.clamp(-1, 1) * 0.5 + 0.5) * 255)
            .clamp(0, 255).type(torch.uint8))


def compute_cyclegan_fid(generator, source_loader, target_loader, device):
    """
    Calcula el FID entre las imágenes generadas por `generator` aplicado a
    las imágenes del dominio fuente y las imágenes reales del dominio
    objetivo.

    Parameters
    ----------
    generator : nn.Module
        G_AB para FID(G_AB(testA), testB), o G_BA para FID(G_BA(testB), testA).
    source_loader : DataLoader
        Loader del dominio fuente (proporciona los inputs al generador).
        Debe iterar sobre la clave 'A' o 'B' según corresponda.
    target_loader : DataLoader
        Loader del dominio objetivo (imágenes reales para el FID).
    device : torch.device
    """
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    generator.eval()

    with torch.no_grad():
        # Imágenes reales del dominio objetivo
        for batch in target_loader:
            real = batch.to(device)
            fid_metric.update(tensor_to_uint8(real), real=True)

        # Imágenes generadas a partir del dominio fuente
        for batch in source_loader:
            src = batch.to(device)
            fake = generator(src)
            fid_metric.update(tensor_to_uint8(fake), real=False)

    fid_value = fid_metric.compute().item()
    generator.train()
    return fid_value


class SingleDomainLoader:
    """Wrapper que extrae solo un dominio del ImageDataset."""

    def __init__(self, dataset, domain, batch_size, num_workers=2):
        self.dataset = dataset
        self.domain = domain  # 'A' o 'B'
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)
        for batch in loader:
            yield batch[self.domain]


testA_loader = SingleDomainLoader(test_dataset, 'A', FID_BATCH_SIZE)
testB_loader = SingleDomainLoader(test_dataset, 'B', FID_BATCH_SIZE)


# ============================================================
# 5. BUCLE DE ENTRENAMIENTO
# ============================================================
print("Comenzando bucle de entrenamiento...", flush=True)

# Trackers para guardar pesos solo cuando mejora el FID
best_fid_AB = float('inf')
best_fid_BA = float('inf')
best_epoch_AB = -1
best_epoch_BA = -1

# Histórico de métricas (una fila por época)
history = {
    'epoch': [], 'loss_G': [], 'loss_D_A': [], 'loss_D_B': [],
    'loss_GAN': [], 'loss_cycle': [], 'loss_identity': [],
    'fid_AB': [], 'fid_BA': [], 'lr': []
}

for epoch in range(EPOCHS):

    # Acumuladores de pérdidas a nivel de época
    epoch_loss_G = 0.0
    epoch_loss_D_A = 0.0
    epoch_loss_D_B = 0.0
    epoch_loss_GAN = 0.0
    epoch_loss_cycle = 0.0
    epoch_loss_identity = 0.0
    n_batches = 0

    for i, batch in enumerate(train_loader):

        real_A = batch['A'].to(DEVICE)
        real_B = batch['B'].to(DEVICE)

        valid = torch.ones(real_A.size(0), *D_A(real_A).shape[1:],
                           requires_grad=False).to(DEVICE)
        fake = torch.zeros(real_A.size(0), *D_A(real_A).shape[1:],
                           requires_grad=False).to(DEVICE)

        
        # Entrenar Generadores
        
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()

        # Identity loss: G_BA(A) aprox A,  G_AB(B) aprox B
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = loss_id_A + loss_id_B          

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = loss_GAN_AB + loss_GAN_BA           

        # Cycle-consistency loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B       

        # Pérdida total del generador 
        loss_G = (loss_GAN
                  + LAMBDA_CYCLE * loss_cycle
                  + LAMBDA_IDENTITY * loss_identity)
        loss_G.backward()
        optimizer_G.step()

        
        # Entrenar Discriminadores
        
        # D_A
        optimizer_D_A.zero_grad()
        loss_real_A = criterion_GAN(D_A(real_A), valid)
        fake_A_replay = fake_A_buffer.push_and_pop(fake_A)
        loss_fake_A = criterion_GAN(D_A(fake_A_replay.detach()), fake)
        loss_D_A = (loss_real_A + loss_fake_A) / 2     # promedio estándar
        loss_D_A.backward()
        optimizer_D_A.step()

        # D_B
        optimizer_D_B.zero_grad()
        loss_real_B = criterion_GAN(D_B(real_B), valid)
        fake_B_replay = fake_B_buffer.push_and_pop(fake_B)
        loss_fake_B = criterion_GAN(D_B(fake_B_replay.detach()), fake)
        loss_D_B = (loss_real_B + loss_fake_B) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        # Acumular
        epoch_loss_G += loss_G.item()
        epoch_loss_D_A += loss_D_A.item()
        epoch_loss_D_B += loss_D_B.item()
        epoch_loss_GAN += loss_GAN.item()
        epoch_loss_cycle += loss_cycle.item()
        epoch_loss_identity += loss_identity.item()
        n_batches += 1

        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}] "
                  f"(Adv: {loss_GAN.item():.4f}, "
                  f"Cycle: {loss_cycle.item():.4f}, "
                  f"Id: {loss_identity.item():.4f})", flush=True)

    
    # Fin de la época
    
    avg_loss_G = epoch_loss_G / n_batches
    avg_loss_D_A = epoch_loss_D_A / n_batches
    avg_loss_D_B = epoch_loss_D_B / n_batches
    avg_loss_GAN = epoch_loss_GAN / n_batches
    avg_loss_cycle = epoch_loss_cycle / n_batches
    avg_loss_identity = epoch_loss_identity / n_batches
    current_lr = optimizer_G.param_groups[0]['lr']

    
    # FID
    
    fid_AB = np.nan
    fid_BA = np.nan
    is_fid_epoch = ((epoch + 1) % FID_FREQ == 0
                    or epoch == 0
                    or epoch == EPOCHS - 1)
    if is_fid_epoch:
        print(f"  Calculando FID en época {epoch+1}...", flush=True)
        fid_AB = compute_cyclegan_fid(G_AB, testA_loader, testB_loader, DEVICE)
        fid_BA = compute_cyclegan_fid(G_BA, testB_loader, testA_loader, DEVICE)
        print(f"  FID(A→B) = {fid_AB:.2f} | FID(B→A) = {fid_BA:.2f}",
              flush=True)

        # Guardar pesos solo si el FID mejora respecto al mejor visto
        if fid_AB < best_fid_AB:
            best_fid_AB = fid_AB
            best_epoch_AB = epoch + 1
            torch.save(G_AB.state_dict(),
                       f"{EXPERIMENT_DIR}/models/G_AB_best.pth")
            torch.save(D_B.state_dict(),
                       f"{EXPERIMENT_DIR}/models/D_B_best.pth")
            print(f"  [BEST A→B] FID={fid_AB:.2f} en época {epoch+1}, "
                  f"pesos guardados.", flush=True)

        if fid_BA < best_fid_BA:
            best_fid_BA = fid_BA
            best_epoch_BA = epoch + 1
            torch.save(G_BA.state_dict(),
                       f"{EXPERIMENT_DIR}/models/G_BA_best.pth")
            torch.save(D_A.state_dict(),
                       f"{EXPERIMENT_DIR}/models/D_A_best.pth")
            print(f"  [BEST B→A] FID={fid_BA:.2f} en época {epoch+1}, "
                  f"pesos guardados.", flush=True)

    # Guardar histórico
    history['epoch'].append(epoch + 1)
    history['loss_G'].append(avg_loss_G)
    history['loss_D_A'].append(avg_loss_D_A)
    history['loss_D_B'].append(avg_loss_D_B)
    history['loss_GAN'].append(avg_loss_GAN)
    history['loss_cycle'].append(avg_loss_cycle)
    history['loss_identity'].append(avg_loss_identity)
    history['fid_AB'].append(fid_AB)
    history['fid_BA'].append(fid_BA)
    history['lr'].append(current_lr)

    # Guardar CSV (sobrescribir cada época)
    pd.DataFrame(history).to_csv(
        f"{EXPERIMENT_DIR}/logs/metrics_cyclegan.csv", index=False)

    
    # Step de los schedulers
    
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    
    # Imágenes de control
    
    if (epoch + 1) % SAVE_IMG_FREQ == 0 or epoch == 0:
        with torch.no_grad():
            G_AB.eval()
            G_BA.eval()

            test_A_path = os.path.join(
                DATASET_DIR, 'testA',
                sorted(os.listdir(os.path.join(DATASET_DIR, 'testA')))[0])
            test_B_path = os.path.join(
                DATASET_DIR, 'testB',
                sorted(os.listdir(os.path.join(DATASET_DIR, 'testB')))[0])

            single_test_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            img_real_A = single_test_transform(
                Image.open(test_A_path).convert('RGB')).unsqueeze(0).to(DEVICE)
            img_real_B = single_test_transform(
                Image.open(test_B_path).convert('RGB')).unsqueeze(0).to(DEVICE)

            img_fake_B = G_AB(img_real_A)
            img_fake_A = G_BA(img_real_B)
            img_recov_A = G_BA(img_fake_B)
            img_recov_B = G_AB(img_fake_A)

            image_grid = torch.cat((img_real_A, img_fake_B, img_recov_A,
                                    img_real_B, img_fake_A, img_recov_B), 0)
            image_grid = image_grid * 0.5 + 0.5

            save_path = f"{EXPERIMENT_DIR}/images/epoch_{epoch+1}.png"
            save_image(image_grid, save_path, nrow=3, normalize=False)
            print(f"--> Imagen de control guardada: {save_path}", flush=True)


# ============================================================
# 6. GUARDADO FINAL Y PLOTS
# ============================================================
print("Entrenamiento finalizado. Guardando modelos finales...", flush=True)
torch.save(G_AB.state_dict(), f"{EXPERIMENT_DIR}/models/G_AB_final.pth")
torch.save(G_BA.state_dict(), f"{EXPERIMENT_DIR}/models/G_BA_final.pth")
torch.save(D_A.state_dict(), f"{EXPERIMENT_DIR}/models/D_A_final.pth")
torch.save(D_B.state_dict(), f"{EXPERIMENT_DIR}/models/D_B_final.pth")

# DataFrame final
df = pd.DataFrame(history)
df.to_csv(f"{EXPERIMENT_DIR}/logs/metrics_cyclegan.csv", index=False)
print(f"CSV final guardado: {EXPERIMENT_DIR}/logs/metrics_cyclegan.csv",
      flush=True)



# Plots

def save_plot(x, ys, labels, colors, title, xlabel, ylabel, filepath,
              markers=None, stds=None):
    """Helper de ploteo (mismo estilo que utils.save_plot del Cap. 5)."""
    plt.figure(figsize=(10, 6))
    for idx, y in enumerate(ys):
        marker = markers[idx] if markers else None
        plt.plot(x, y, label=labels[idx], color=colors[idx], marker=marker)
        if stds is not None and stds[idx] is not None:
            plt.fill_between(x,
                             np.array(y) - np.array(stds[idx]),
                             np.array(y) + np.array(stds[idx]),
                             color=colors[idx], alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# 1) Curva conjunta loss_G y loss_D (suma de los dos discriminadores)
df['loss_D_total'] = df['loss_D_A'] + df['loss_D_B']
save_plot(
    x=df['epoch'],
    ys=[df['loss_G'], df['loss_D_total']],
    labels=['Pérdida del Generador (suma)', 'Pérdida de los Discriminadores'],
    colors=['blue', 'orange'],
    title='Curvas de Aprendizaje CycleGAN - apple2orange',
    xlabel='Épocas', ylabel='Pérdida',
    filepath=f"{EXPERIMENT_DIR}/plots/training_losses.png"
)

# 2) Descomposición de la pérdida del generador
save_plot(
    x=df['epoch'],
    ys=[df['loss_GAN'], df['loss_cycle'], df['loss_identity']],
    labels=['Adversarial', 'Cycle (×1)', 'Identity (×1)'],
    colors=['red', 'green', 'purple'],
    title='Componentes de la Pérdida del Generador (sin λ aplicado)',
    xlabel='Épocas', ylabel='Pérdida',
    filepath=f"{EXPERIMENT_DIR}/plots/generator_components.png"
)

# 3) Curvas de los dos discriminadores por separado
save_plot(
    x=df['epoch'],
    ys=[df['loss_D_A'], df['loss_D_B']],
    labels=['D_A (manzanas)', 'D_B (naranjas)'],
    colors=['darkorange', 'gold'],
    title='Pérdidas de los Discriminadores',
    xlabel='Épocas', ylabel='Pérdida',
    filepath=f"{EXPERIMENT_DIR}/plots/discriminator_losses.png"
)

# 4) FID en ambos sentidos (solo épocas con FID)
df_fid = df.dropna(subset=['fid_AB', 'fid_BA'])
if len(df_fid) > 0:
    save_plot(
        x=df_fid['epoch'],
        ys=[df_fid['fid_AB'], df_fid['fid_BA']],
        labels=['FID(G_AB(A) → B): manzana → naranja',
                'FID(G_BA(B) → A): naranja → manzana'],
        colors=['firebrick', 'royalblue'],
        title='Evolución del FID en CycleGAN - apple2orange',
        xlabel='Épocas', ylabel='FID',
        filepath=f"{EXPERIMENT_DIR}/plots/fid_evolution.png",
        markers=['o', 's']
    )

# 5) Learning rate
save_plot(
    x=df['epoch'], ys=[df['lr']],
    labels=['Learning rate'], colors=['black'],
    title='Programación del Learning Rate',
    xlabel='Épocas', ylabel='LR',
    filepath=f"{EXPERIMENT_DIR}/plots/lr_schedule.png"
)

print(f"Plots guardados en {EXPERIMENT_DIR}/plots/", flush=True)
print(f"¡Hecho! Todo guardado en {EXPERIMENT_DIR}", flush=True)


# Resumen final

print("\n=== RESUMEN ===", flush=True)
print(f"FID inicial    A→B: {df['fid_AB'].iloc[0]:.2f} | "
      f"B→A: {df['fid_BA'].iloc[0]:.2f}", flush=True)
print(f"FID mejor      A→B: {best_fid_AB:.2f} (época {best_epoch_AB}) | "
      f"B→A: {best_fid_BA:.2f} (época {best_epoch_BA})", flush=True)
print(f"FID final      A→B: {df['fid_AB'].iloc[-1]:.2f} | "
      f"B→A: {df['fid_BA'].iloc[-1]:.2f}", flush=True)
print(f"\nPesos guardados en {EXPERIMENT_DIR}/models/:", flush=True)
print(f"  Mejor A→B: G_AB_best.pth, D_B_best.pth (época {best_epoch_AB})",
      flush=True)
print(f"  Mejor B→A: G_BA_best.pth, D_A_best.pth (época {best_epoch_BA})",
      flush=True)
print(f"  Final:     G_AB_final.pth, G_BA_final.pth, "
      f"D_A_final.pth, D_B_final.pth", flush=True)
