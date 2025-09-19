import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from self_attention import SelfAttention

class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(Discriminator, self).__init__()
        channels, height, width = img_shape
        self.label_embedding = nn.Embedding(num_classes, num_classes)  # one-hot embedding

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(channels + num_classes, 64, kernel_size=4, stride=2, padding=1)),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # 64x64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # 32x32
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(256),  # увага на середньому рівні ознак (оптимальний компроміс)

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),  # 16x16
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0))  # → 13x13
        )

    def forward(self, img, labels):
        B, _, H, W = img.size()
        label_embeddings = self.label_embedding(labels)  # (B, C)
        label_map = label_embeddings.view(B, -1, 1, 1).expand(B, -1, H, W)  # → (B, C, H, W)
        d_in = torch.cat((img, label_map), dim=1)

        out = self.model(d_in)
        return out.view(out.size(0), -1).mean(1, keepdim=True)  # scalar output
