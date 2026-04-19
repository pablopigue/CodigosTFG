import os
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
plt.switch_backend('agg') 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros estándar de CycleGAN
EPOCHS = 100             
BATCH_SIZE = 1           
LR = 0.0002
BETA1 = 0.5
LAMBDA_CYCLE = 10.0      
LAMBDA_IDENTITY = 5.0    
SAVE_IMG_FREQ = 5        
SAVE_MODEL_FREQ = 5      

# Rutas
EXPERIMENT_DIR = "/mnt/homeGPU/pablomarpa/CodigosTFG/tfg_cyclegan_apple2orange"
DATASET_DIR = "./apple2orange" 

# Crear estructura de carpetas
os.makedirs(f"{EXPERIMENT_DIR}/images", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/models", exist_ok=True)

print(f"Iniciando entrenamiento CycleGAN en: {DEVICE}", flush=True)

# ==============================================================================
# 1. DATASET PERSONALIZADO
# ==============================================================================
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms_
        path_A = os.path.join(root, f'{mode}A')
        path_B = os.path.join(root, f'{mode}B')
        
        self.files_A = sorted([os.path.join(path_A, x) for x in os.listdir(path_A) if x.endswith(('.jpg', '.png'))])
        self.files_B = sorted([os.path.join(path_B, x) for x in os.listdir(path_B) if x.endswith(('.jpg', '.png'))])

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

transforms_ = transforms.Compose([
    transforms.Resize(int(256 * 1.12), Image.BICUBIC),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

print("Cargando dataset...", flush=True)
train_dataset = ImageDataset(DATASET_DIR, transforms_=transforms_, mode='train')
# AJUSTE: Bajamos num_workers a 2 para evitar el warning y posible lentitud
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ==============================================================================
# 2. MODELOS 
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
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
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(channels, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features // 2

        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, channels, 7),
                    nn.Tanh() ] 

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# --- CORRECCIÓN IMPORTANTE AQUÍ ---
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        # InstanceNorm2d por defecto no tiene weights (affine=False), hay que comprobarlo
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

# ==============================================================================
# 3. PREPARACIÓN 
# ==============================================================================
input_shape = (3, 256, 256)

G_AB = GeneratorResNet(input_shape, num_residual_blocks=9).to(DEVICE) 
G_BA = GeneratorResNet(input_shape, num_residual_blocks=9).to(DEVICE) 
D_A = Discriminator(input_shape).to(DEVICE) 
D_B = Discriminator(input_shape).to(DEVICE) 

# Aplicar pesos corregidos
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

criterion_GAN = torch.nn.MSELoss() 
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(BETA1, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(BETA1, 0.999))

class ReplayBuffer:
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

# ==============================================================================
# 4. BUCLE PRINCIPAL
# ==============================================================================
print("Comenzando bucle de entrenamiento...", flush=True)

for epoch in range(EPOCHS):
    for i, batch in enumerate(train_loader):
        
        real_A = batch['A'].to(DEVICE)
        real_B = batch['B'].to(DEVICE)

        valid = torch.ones(real_A.size(0), *D_A(real_A).shape[1:], requires_grad=False).to(DEVICE)
        fake = torch.zeros(real_A.size(0), *D_A(real_A).shape[1:], requires_grad=False).to(DEVICE)

        # ----------------------
        #  Entrenar Generadores
        # ----------------------
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()

        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid) 
        
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid) 
        
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        recov_A = G_BA(fake_B) 
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        
        recov_B = G_AB(fake_A) 
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        loss_G = loss_GAN + LAMBDA_CYCLE * loss_cycle + LAMBDA_IDENTITY * loss_identity
        loss_G.backward()
        optimizer_G.step()

        # -------------------------
        #  Entrenar Discriminadores
        # -------------------------
        optimizer_D_A.zero_grad()
        loss_real = criterion_GAN(D_A(real_A), valid)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_real = criterion_GAN(D_B(real_B), valid)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}] (Adv: {loss_GAN.item():.4f}, Cycle: {loss_cycle.item():.4f})", flush=True)

    # ---------------------------------------------
    #  Guardar Imágenes 
    # ---------------------------------------------
    if (epoch + 1) % SAVE_IMG_FREQ == 0 or epoch == 0:
        with torch.no_grad():
            G_AB.eval()
            G_BA.eval()
            
            test_A_path = os.path.join(DATASET_DIR, 'testA', sorted(os.listdir(os.path.join(DATASET_DIR, 'testA')))[0])
            test_B_path = os.path.join(DATASET_DIR, 'testB', sorted(os.listdir(os.path.join(DATASET_DIR, 'testB')))[0])
            
            test_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            img_real_A = test_transform(Image.open(test_A_path).convert('RGB')).unsqueeze(0).to(DEVICE)
            img_real_B = test_transform(Image.open(test_B_path).convert('RGB')).unsqueeze(0).to(DEVICE)
            
            img_fake_B = G_AB(img_real_A)
            img_fake_A = G_BA(img_real_B)
            img_recov_A = G_BA(img_fake_B)
            img_recov_B = G_AB(img_fake_A)
            
            image_grid = torch.cat((img_real_A, img_fake_B, img_recov_A, 
                                    img_real_B, img_fake_A, img_recov_B), 0)
            
            image_grid = image_grid * 0.5 + 0.5
            
            save_path = f"{EXPERIMENT_DIR}/images/epoch_{epoch+1}.png"
            save_image(image_grid, save_path, nrow=3, normalize=False)
            print(f"--> Guardada imagen de control: {save_path}", flush=True)

    # ---------------------------------------------
    #  Guardar Checkpoints
    # ---------------------------------------------
    if (epoch + 1) % SAVE_MODEL_FREQ == 0:
        torch.save(G_AB.state_dict(), f"{EXPERIMENT_DIR}/models/G_AB_{epoch+1}.pth")
        torch.save(G_BA.state_dict(), f"{EXPERIMENT_DIR}/models/G_BA_{epoch+1}.pth")
        torch.save(D_A.state_dict(), f"{EXPERIMENT_DIR}/models/D_A_{epoch+1}.pth")
        torch.save(D_B.state_dict(), f"{EXPERIMENT_DIR}/models/D_B_{epoch+1}.pth")
        print(f"--> Checkpoints guardados en época {epoch+1}", flush=True)

print("Entrenamiento finalizado. Guardando modelos finales...", flush=True)
torch.save(G_AB.state_dict(), f"{EXPERIMENT_DIR}/models/G_AB_final.pth")
torch.save(G_BA.state_dict(), f"{EXPERIMENT_DIR}/models/G_BA_final.pth")
torch.save(D_A.state_dict(), f"{EXPERIMENT_DIR}/models/D_A_final.pth")
torch.save(D_B.state_dict(), f"{EXPERIMENT_DIR}/models/D_B_final.pth")
print(f"¡Hecho! Todo guardado en {EXPERIMENT_DIR}", flush=True)