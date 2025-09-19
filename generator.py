import torch
import torch.nn as nn
from self_attention import SelfAttention

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels, img_size=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.init_size = img_size // 4  # 64 для 256x256

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * self.init_size ** 2),
            nn.ReLU(True)
        )

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),

            SelfAttention(128),                # ← attention на 64x64
            nn.Upsample(scale_factor=2),       # → 128x128
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            SelfAttention(128),                # ← доданий attention на 128x128
            nn.Upsample(scale_factor=2),       # → 256x256
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, img_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((noise, label_input), dim=1)
        out = self.fc(x)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.model(out)
