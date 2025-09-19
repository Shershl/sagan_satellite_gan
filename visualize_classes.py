import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
num_classes = 4

generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

def generate_and_plot(generator, num_images=4):
    fig, axs = plt.subplots(num_classes, num_images, figsize=(num_images * 2, num_classes * 2))
    for cls in range(num_classes):
        for i in range(num_images):
            z = torch.randn(1, latent_dim).to(device)
            y = torch.tensor([cls]).to(device)
            with torch.no_grad():
                out = generator(z, y).squeeze().cpu()
            img = transforms.ToPILImage()(out.clamp(-1, 1) * 0.5 + 0.5)
            axs[cls, i].imshow(img)
            axs[cls, i].axis('off')
        axs[cls, 0].set_ylabel(f"Клас {cls}")
    plt.tight_layout()
    plt.savefig("generated_by_class.png")

generate_and_plot(generator)
