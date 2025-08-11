
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class PigCoughDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use 'id' and 'genus' column names instead of positional indexing
        img_name = os.path.join(self.image_dir, f"{self.data.loc[idx, 'id']}.png")
        image = Image.open(img_name).convert('RGB')  # Ensure 3-channel RGB image

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # Default transformation

        label = int(self.data.loc[idx, 'genus'])  # Convert label to integer
        return image, label


