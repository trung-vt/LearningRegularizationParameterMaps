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


def convert_to_tensor_4D(image_in):
    # xf = []
    # xf.append(image_numpy)
    # xf = np.stack(xf, axis=-1)
    # xf = torch.tensor(xf, dtype=torch.float)
    image_numpy = convert_to_numpy(image_in) # in case image_in is a PIL.Image or something else. Note that converting a ndarray to a ndarray is a no-op.
    xf = torch.tensor(image_numpy, dtype=torch.float)
    xf = xf.unsqueeze(0)
    xf = xf.unsqueeze(-1)
    xf = xf / 255 # Normalise from [0, 255] to [0, 1]
    return xf

def convert_to_PIL(image_tensor_4D):
    img = image_tensor_4D.squeeze(0).squeeze(-1)
    img = img * 255
    img = img.detach().cpu().numpy().astype(np.uint8)
    img = Image.fromarray(img)
    return img


def get_variable_noise(sigma_min, sigma_max, manual_seed=None):
    if manual_seed is not None: torch.manual_seed(manual_seed)
    return sigma_min + torch.rand(1) * (sigma_max - sigma_min)


def add_noise(x_original: torch.tensor, sigma, manual_seed=None) -> torch.tensor:
    """
    Add data independent Gaussian noise to the image.
    The noise follows a normal distribution with mean 0 and standard deviation sigma.
    If manual_seed is not None, set the random seed to manual_seed for reproducibility.
    
    Parameters
    ----------
    
    """
    if manual_seed is not None: torch.manual_seed(manual_seed)
        
    # std = torch.std(x_original)
    # mu = torch.mean(x_original)

    # x_centred = (x_original  - mu) / std
    
    z = torch.randn(x_original.shape, dtype = x_original.dtype) # Standard normal (0, 1)
    
    noise = z * sigma # Normal (0, sigma^2) --> noise (data independent)

    # x_centred += noise

    # xnoise = std * x_centred + mu # add data dependent noise to the original image???
    # xnoise = xf + std * y         # equivalent to above
    
    x_noisy = x_original + noise # add (data independent) noise to the original image
    
    x_noisy = torch.clamp(x_noisy, 0, 1) # clip to [0, 1]

    # del std, mu, x_centred
    del z, noise

    return x_noisy


# def add_noise_PIL(image: Image, sigma, manual_seed=None) -> Image:
#     image_tensor = torch.tensor(np.asarray(image), dtype=torch.float) / 255
#     image_noisy_tensor = add_noise(image_tensor, sigma, manual_seed)
#     image_noisy_tensor *= 255
#     image_noisy_tensor = torch.clamp(image_noisy_tensor, 0, 255)
#     image_noisy = Image.fromarray(image_noisy_tensor.numpy().astype(np.uint8))
#     return image_noisy