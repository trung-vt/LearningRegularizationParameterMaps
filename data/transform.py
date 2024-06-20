import torch
from PIL import Image # for image manipulation
import numpy as np


def crop_and_resize(img:Image, size:int=256):
    """
    Crop to a square (centered) and resize the image.
    
    Parameters
    ----------
    img : PIL.Image
        Image to crop and resize.
    
    size : int
        Size of the square image. Default is 256 by 256.
    
    Returns
    -------
    PIL.Image
        Cropped and resized image.
    """
    # crop the image to a square, centered
    width, height = img.size
    new_width = min(width, height)
    new_height = min(width, height)
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))
    
    # resize the image
    img = img.resize((size, size))
    
    return img


def convert_to_grayscale(image: Image) -> Image:
    return image.convert('L')


def convert_to_numpy(image):
    image_data = np.asarray(image)
    return image_data


def convert_to_tensor_4D(image_numpy):
    # xf = []
    # xf.append(image_numpy)
    # xf = np.stack(xf, axis=-1)
    # xf = torch.tensor(xf, dtype=torch.float)
    xf = torch.tensor(image_numpy, dtype=torch.float)
    xf = xf.unsqueeze(0)
    xf = xf.unsqueeze(-1)
    xf = xf / 255 # Normalise from [0, 255] to [0, 1]
    return xf


def get_variable_noise(sigma_min, sigma_max):
    return sigma_min + torch.rand(1) * (sigma_max - sigma_min)


def add_noise(xf: torch.tensor, sigma) -> torch.tensor:
    std = torch.std(xf)
    mu = torch.mean(xf)

    x_centred = (xf  - mu) / std

    x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)

    xnoise = std * x_centred + mu

    del std, mu, x_centred

    return xnoise