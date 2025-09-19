# ОНОВЛЕНИЙ Train.py з урахуванням Hinge Loss, heatmaps, SN, attention

import os
import torch
import csv
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image
from dataset_loader import SatelliteDataset
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator

# Параметри
z_dim = 128
img_shape = (3, 256, 256)
batch_size = 32
epochs = 200
start_epoch = 0
num_classes = 4

log_file = "training_log.csv"
image_dir = "generated_samples"
checkpoint_dir = "checkpoints"
heatmap_dir = "heatmaps"

# Створення директорій
os.makedirs(image_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(heatmap_dir, exist_ok=True)

# Датасет
dataset = SatelliteDataset(
    annotations_file="annotations.csv",
    img_dir="C:/artificial intelligent systems/Dataset/military",
    image_size=256
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Моделі
netG = Generator(z_dim, num_classes, img_shape[0]).to(device)
netD = Discriminator(img_shape, num_classes).to(device)

# Оптимізатори
optimizer_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
scaler_G = GradScaler()
scaler_D = GradScaler()

# Лог-файл
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "loss_D", "loss_G"])

# Навчання
for epoch in range(start_epoch, epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        # === Генератор ===
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), z_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (imgs.size(0),), device=device)

        with autocast():
            gen_imgs = netG(z, gen_labels)
            fake_validity = netD(gen_imgs, gen_labels)
            g_loss = -torch.mean(fake_validity)  # Hinge loss

        scaler_G.scale(g_loss).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        # === Дискримінатор ===
        optimizer_D.zero_grad()
        with autocast():
            real_validity = netD(imgs, labels)
            fake_validity = netD(gen_imgs.detach(), gen_labels)
            d_loss = torch.mean(torch.relu(1. - real_validity)) + \
                     torch.mean(torch.relu(1. + fake_validity))

        scaler_D.scale(d_loss).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()

        # === Логування ===
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, i, d_loss.item(), g_loss.item()])

        print(f"[Epoch {epoch+1}/{epochs}] Batch {i}/{len(dataloader)} | Loss_D: {d_loss.item():.4f} | Loss_G: {g_loss.item():.4f}")

    # === Збереження кожні 10 епох ===
    if (epoch + 1) % 10 == 0:
        torch.save(netG.state_dict(), f"{checkpoint_dir}/G_epoch_{epoch + 1}.pth")
        torch.save(netD.state_dict(), f"{checkpoint_dir}/D_epoch_{epoch + 1}.pth")

        with torch.no_grad():
            z = torch.randn(16, z_dim, device=device)
            sample_labels = torch.randint(0, num_classes, (16,), device=device)
            gen_samples = netG(z, sample_labels)
            save_image(gen_samples.data, f"{image_dir}/epoch_{epoch + 1}.png", nrow=4, normalize=True)

            # === Збереження heatmap attention ===
            latent = netG.fc(torch.cat((z[:1], netG.label_emb(sample_labels[:1])), dim=1))
            latent = latent.view(1, 128, 64, 64)
            attention_layer = netG.model[0]  # перший SelfAttention
            _, heatmap = attention_layer(latent, return_attention=True)
            save_image(heatmap.mean(1).unsqueeze(1), f"{heatmap_dir}/epoch_{epoch + 1}.png", normalize=True)

        print(f"Checkpoint, images and heatmap saved at epoch {epoch + 1}")
