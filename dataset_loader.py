import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SatelliteDataset(Dataset):
    def __init__(self, annotations_file, img_dir, image_size=256, transform=None, limit_size=None):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.df = self.df[['image_path', 'object_type']]

        if limit_size:
            self.df = self.df.sample(n=limit_size, random_state=42).reset_index(drop=True)

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_dir, row['image_path'])  # використовує img_dir з аргументу
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = int(row['object_type'])  # клас як int
        return image, label
