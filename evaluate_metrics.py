import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
from scipy import linalg
import lpips
from tqdm import tqdm
from generator import Generator  # імпорт вашого генератора

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Налаштування ===
num_samples = 100  # по скільки зображень генерується
image_size = 256
latent_dim = 128
num_classes = 4
real_path = "real_samples"  # шлях до справжніх зображень
model_path = "generator.pth"

# === Завантаження генератора ===
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# === Генерація зображень ===
def generate_images(generator, num_samples):
    images = []
    for _ in range(num_samples):
        z = torch.randn(1, latent_dim).to(device)
        y = torch.randint(0, num_classes, (1,)).to(device)
        fake = generator(z, y)
        fake = fake.squeeze(0).detach().cpu()
        img = transforms.ToPILImage()(fake.clamp(-1, 1) * 0.5 + 0.5)
        images.append(img)
    return images

# === Завантаження реальних зображень ===
def load_real_images(path, num_samples):
    dataset = datasets.ImageFolder(path, transform=transforms.ToTensor())
    real_images = []
    for i in range(min(num_samples, len(dataset))):
        img, _ = dataset[i]
        real_images.append(transforms.ToPILImage()(img))
    return real_images

# === Функція ознак для FID ===
def get_inception_features(images, model):
    model.eval()
    features = []
    with torch.no_grad():
        for img in images:
            img = transforms.Resize((299, 299))(img)
            img = transforms.ToTensor()(img).unsqueeze(0).to(device)
            img = transforms.Normalize([0.5]*3, [0.5]*3)(img)
            feat = model(img)
            features.append(feat.squeeze().cpu().numpy())
    return np.array(features)

# === Обчислення FID ===
def calculate_fid(features1, features2):
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# === Обчислення LPIPS ===
def calculate_lpips(imgs1, imgs2):
    loss_fn = lpips.LPIPS(net='alex').to(device)
    scores = []
    for im1, im2 in zip(imgs1, imgs2):
        t1 = transforms.ToTensor()(im1).unsqueeze(0).to(device)
        t2 = transforms.ToTensor()(im2).unsqueeze(0).to(device)
        scores.append(loss_fn(t1, t2).item())
    return np.mean(scores)

# === Основний блок ===
print("🔄 Генерація зображень...")
fake_images = generate_images(generator, num_samples)
print("✅ Завантаження реальних зображень...")
real_images = load_real_images(real_path, num_samples)

print("📊 Обчислення FID та LPIPS...")
inception = inception_v3(pretrained=True, transform_input=False).to(device)
features_fake = get_inception_features(fake_images, inception)
features_real = get_inception_features(real_images, inception)

fid_score = calculate_fid(features_real, features_fake)
lpips_score = calculate_lpips(fake_images, real_images)

print(f"\n📈 Результати:")
print(f"FID: {fid_score:.2f}")
print(f"LPIPS: {lpips_score:.4f}")
