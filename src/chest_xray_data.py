import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class ChestXRayDataset(Dataset):
    def __init__(self, images_folder, transform=None, max_pixel_value=255.0, sigma_min=0.1, sigma_max=0.5, num_images=10, device="cuda"):
        self.images_folder = images_folder
        self.transform = transform
        self.images = os.listdir(images_folder)

        # Only keep the files with .jpeg extension
        self.images = [img for img in self.images if img.endswith(".jpeg")]

        # Limit the number of images for testing
        self.images = self.images[:num_images]
        
        self.max_pixel_value = max_pixel_value
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.images[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = self.default_transform(image)
        
        noisy_image = self.add_noise(image, self.get_variable_noise())
        return noisy_image.to(self.device), image.to(self.device)
    
    def default_transform(self, image):
        image = self.convert_jpeg_to_tensor(image)
        image = image / self.max_pixel_value # Normalise the pixel values to [0, 1]
        image = self.extract_square(image, 120)
        if len(image.shape) == 2:
            image = image.unsqueeze(0) # Add channel dimension to the front. From (64, 64) to (1, 64, 64)

        # TODO: Add channels for testing with known working autoencoder code only. Remove this later
        # If single channel, convert to 3 channels by repeating the same channel 3 times
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        assert len(image.shape) == 3, f"Expected 3D image tensor, got {image.shape}"

        # # TODO: Add time dimension for working with legacy dynamic image code only. Remove this later
        # image = image.unsqueeze(-1)

        return image
    
    def extract_square(self, image, size):
        # Assuming the image is larger than size x size
        h, w = image.shape[0], image.shape[1]
        top = (h - size) // 2
        left = (w - size) // 2
        bottom = top + size
        right = left + size
        image = image[top:bottom, left:right]
        assert image.shape == (size, size), f"Expected image shape ({size}, {size}), got {image.shape}"
        return image
    
    def convert_jpeg_to_tensor(self, image):
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32)
        return image
    
    def get_variable_noise(self):
        sigma_min, sigma_max = self.sigma_min, self.sigma_max
        return sigma_min + torch.rand(1) * (sigma_max - sigma_min)

    def add_noise(self, xf: torch.tensor, sigma) -> torch.tensor:
        std = torch.std(xf)
        mu = torch.mean(xf)
        x_centred = (xf  - mu) / std
        x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)
        xnoise = std * x_centred + mu
        return xnoise
    

def get_data_loader(
        images_folder:str, device, manual_seed=42, batch_size=1, num_workers=4, shuffle=False, 
        sigma_min=0.1, sigma_max=0.5,
        transform=None,
        num_images=10):
    xray_dataset = ChestXRayDataset(images_folder=images_folder, transform=transform, sigma_min=sigma_min, sigma_max=sigma_max, num_images=num_images, device=device)

    data_loader = torch.utils.data.DataLoader(xray_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=torch.Generator(device=device).manual_seed(manual_seed))

    return data_loader


def transform_train(image):
    image = image / 255.0
    return image


def get_data_loaders_all(config, num_images=10):
    bd = config.data.base_dir
    de = config.device
    ms = config.data_loader.manual_seed

    bs = config.data_loader.batch_size
    nw = config.data_loader.num_workers
    sf = config.data_loader.shuffle

    sig_min = config.data_loader.noise_sigma.min
    sig_max = config.data_loader.noise_sigma.max

    train_loader = get_data_loader(
        images_folder=f"{bd}/train/NORMAL", device=de, manual_seed=ms, batch_size=bs, num_workers=nw, shuffle=sf,
        sigma_min=sig_min, sigma_max=sig_max, 
        # transform=config.data_loader.train.transform,
        transform=None,
        num_images=num_images
    )

    val_loader = get_data_loader(
        images_folder=f"{bd}/val/NORMAL", device=de, manual_seed=ms, batch_size=bs, num_workers=nw, shuffle=sf, 
        sigma_min=sig_min, sigma_max=sig_max, 
        # transform=config.data_loader.val.transform,
        transform=None,
        num_images=num_images
    )

    test_loader = get_data_loader(
        images_folder=f"{bd}/test/NORMAL", device=de, manual_seed=ms, batch_size=bs, num_workers=nw, shuffle=sf, 
        sigma_min=sig_min, sigma_max=sig_max, 
        # transform=config.data_loader.test.transform,
        transform=None,
        num_images=num_images
    )

    return train_loader, val_loader, test_loader



def test_get_data_loader(config, stage="val", num_images=10):
    bd = config.data.base_dir
    de = config.device
    ms = config.data_loader.manual_seed

    bs = config.data_loader.batch_size
    nw = config.data_loader.num_workers
    sf = config.data_loader.shuffle

    data_loader = get_data_loader(
        images_folder=f"{bd}/{stage}/NORMAL", device=de, manual_seed=ms, batch_size=bs, num_workers=nw, shuffle=sf, 
        transform=None,
        num_images=num_images
    )

    print(f"Number of batches: {len(data_loader)}")

    for i, data in enumerate(data_loader):
        noisy_image, clean_image = data
        print(f"Noisy image {i} shape: {noisy_image.shape}")
        print(f"Clean image {i} shape: {clean_image.shape}")
        if i == 0:
            break