import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in {__file__} ...")
import_torch_start_time = time.time() 
import torch
print(f"Importing torch took {time.time() - import_torch_start_time} seconds")

import os
import random
from PIL import Image
from tqdm import tqdm # progress bar
import numpy as np

def get_a_random_location(im_shape: tuple, patch_shape: tuple) -> tuple:
    im_h, im_w = im_shape
    patch_h, patch_w = patch_shape
    if im_h < patch_h or im_w < patch_w:
        print(f"Image size is smaller than the patch size. Skipping ...")
        return None
    x = random.randint(0, im_w - patch_w)
    y = random.randint(0, im_h - patch_h)
    return x, y


def get_a_patch(input_image: Image, loc: tuple, patch_size=(256, 256)) -> Image:
    im_h, im_w = input_image.size
    patch_h, patch_w = patch_size
    x, y = loc
    if x + patch_w > im_w or y + patch_h > im_h:
        print(f"Patch is out of bounds. Skipping ...")
        return None
    patch = input_image.crop((x, y, x + patch_w, y + patch_h))
    return patch


def get_random_patches(input_image: Image, n_patches: int, patch_size=(256, 256)) -> list:
    im_h, im_w = input_image.size
    patch_h, patch_w = patch_size
    if im_h < patch_h or im_w < patch_w:
        print(f"Image size is smaller than the patch size. Skipping ...")
        return None
    patches = []
    for _ in range(n_patches):
        x, y = get_a_random_location(input_image.size, patch_size)
        patch = get_a_patch(input_image, (x, y), patch_size)
        patches.append(patch)
    
    return patches


# def add_gaussian_noise(image: Image, sigma_min: int, sigma_max: int) -> Image:
#     # print(f"image.size: {image.size}")
#     image = np.array(image, dtype=np.float32) / 255.0
#     # row,col,ch= image.shape
#     col, row = image.shape
#     mean = 0
#     # var = 0.1
#     # sigma = var**0.5
#     sigma = sigma_min + np.random.rand(1) * (sigma_max - sigma_min)
#     # gauss = np.random.normal(mean,sigma,(row,col,ch))
#     gauss = np.random.normal(mean,sigma,(col, row))
#     # gauss = gauss.reshape(row,col,ch)
#     gauss = gauss.reshape(col, row)
#     noisy = image + gauss
#     # noisy = np.clip(noisy, 0, 1)
#     noisy = np.uint8(noisy*255)
#     noisy = Image.fromarray(noisy)
#     return noisy


# def get_variable_noise(sigma_min, sigma_max):
#     return sigma_min + torch.rand(1) * (sigma_max - sigma_min)

# def add_noise(xf: torch.tensor, sigma) -> torch.tensor:
#     std = torch.std(xf)
#     mu = torch.mean(xf)

#     x_centred = (xf  - mu) / std

#     x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)

#     xnoise = std * x_centred + mu

#     return xnoise

# def add_gaussian_noise(image: Image, sigma_min: int, sigma_max: int) -> Image:
#     im_np = np.array(image)
#     im_tensor = torch.tensor(im_np, dtype=torch.float)
#     im_tensor /= 255.0
#     sigma = get_variable_noise(sigma_min, sigma_max)
#     noisy_im_tensor = add_noise(im_tensor, sigma)
#     noisy_im_tensor *= 255.0
#     noisy_im_np = noisy_im_tensor.to("cpu").detach().numpy().astype('uint8')
#     # noisy_im_np = np.clip(noisy_im_np, 0, 255)
#     noisy_image = Image.fromarray(noisy_im_np)
#     return noisy_image


# def add_gaussian_noise(image: Image, sigma_min: int, sigma_max: int) -> Image:
#     image_np = np.array(image)

#     xf = torch.tensor(image_np, dtype=torch.float)

#     xf /= 255.0

#     std = torch.std(xf)
#     mu = torch.mean(xf)

#     x_centred = (xf  - mu) / std

#     sigma = sigma_min + torch.rand(1) * (sigma_max - sigma_min)

#     x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)

#     xnoise = std * x_centred + mu

#     xnoise *= 255.0

#     xnoise_np = xnoise.to("cpu").detach().numpy().astype('uint8')

#     del std, mu, x_centred, xnoise, xf

#     # clip the values to [0, 255]
#     xnoise_np = np.clip(xnoise_np, 0, 255)

#     noisy_image = Image.fromarray(xnoise_np)

#     return noisy_image


def save_patches(patches: list, output_folder: str, output_file: str, min_sigma=0.1, max_sigma=0.5) -> None:
    for i, patch in enumerate(patches):
        folder = f"{output_folder}/{output_file}_{i}"
        os.makedirs(folder, exist_ok=True)

        # patch_channels = patch.split()
        # noisy_patch_channels = []
        # for j, color in enumerate(["R", "G", "B"]):
        #     patch_single_channel = patch_channels[j]
        #     noisy_patch_single_channel = add_gaussian_noise(patch_single_channel, min_sigma, max_sigma)
        #     noisy_patch_channels.append(noisy_patch_single_channel)

        # noisy_patch = Image.merge("RGB", noisy_patch_channels)
        
        noisy_patch = add_gaussian_noise(patch, min_sigma, max_sigma)
        
        patch.save(f"{folder}/clean.png")
        noisy_patch.save(f"{folder}/noisy.png")


def get_files_with_condition(outer_path: str, condition) -> list:
    folders = os.listdir(outer_path)
    return_files = []
    for folder in folders:
        files = os.listdir(f"{outer_path}/{folder}")
        for file in files:
            if condition(file):
                image_file = f"{outer_path}/{folder}/{file}"
                return_files.append(image_file)
    return return_files


def get_sidd_files(outer_path: str, type: str, end: str = "PNG") -> list:
    condition = lambda file: file.endswith(end) and type in file
    return get_files_with_condition(outer_path, condition)


def get_and_save_patches(
        input_path: str, output_path: str, type="GT",  # type="GT" or "NOISY
        n_patches=32, patch_size=(256, 256), min_sigma=0.1, max_sigma=0.5,
        rand_seed=97) -> None:
    
    os.makedirs(output_path, exist_ok=True) # Create output folder if it doesn't exist

    if rand_seed is not None:
        random.seed(rand_seed) # Random seed for reproducibility

    sidd_files = get_sidd_files(input_path, type)
    for i, file in tqdm(enumerate(sidd_files)):
        image = Image.open(file)
        image = image.convert("L") # Convert to grayscale
        patches = get_random_patches(image, n_patches, patch_size)
        output_file = file.split("/")[-1].split(".")[0]
        save_patches(patches, output_path, output_file, min_sigma, max_sigma)
        break
