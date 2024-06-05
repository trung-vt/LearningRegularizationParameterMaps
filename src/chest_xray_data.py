import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class ChestXRayDataset(Dataset):
    def __init__(self, images_folder, transform=None, device="cuda"):
        self.images_folder = images_folder
        self.transform = transform
        self.images = os.listdir(images_folder)
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.images[idx])
        image = Image.open(img_name)
        image = self.convert_jpeg_to_tensor(image)
        if self.transform:
            image = self.transform(image)
        return image
    
    def convert_jpeg_to_tensor(self, image):
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32)
        return image