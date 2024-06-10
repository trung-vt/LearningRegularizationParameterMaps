# import time
# # NOTE: Importing torch the first time will always take a long time!
# print(f"Importing torch in {__file__} ...")
# import_torch_start_time = time.time() 
# import torch
# print(f"Importing torch took {time.time() - import_torch_start_time} seconds")

import os
import random
from PIL import Image
from tqdm import tqdm # progress bar
import numpy as np

def get_a_random_location(im_shape: tuple, patch_shape: tuple) -> tuple:
    im_w, im_h = im_shape
    patch_w, patch_h = patch_shape
    if im_w < patch_w or im_h < patch_h:
        print(f"Image size is smaller than the patch size. Skipping ...")
        return None
    x = random.randint(0, im_w - patch_w)
    y = random.randint(0, im_h - patch_h)
    return x, y


def get_a_patch(input_image: Image, loc: tuple, patch_size=(256, 256)) -> Image:
    im_w, im_h = input_image.size
    patch_w, patch_h = patch_size
    x, y = loc
    if x + patch_w > im_w or y + patch_h > im_h:
        print(f"Patch is out of bounds. Skipping ...")
        return None
    patch = input_image.crop((x, y, x + patch_w, y + patch_h))
    # print(f"x1: {x}, y1: {y}, x2: {x + patch_w}, y2: {y + patch_h}")
    return patch


def get_random_patches(input_image: Image, n_patches: int, patch_size=(256, 256)) -> list:
    im_w, im_h = input_image.size
    patch_w, patch_h = patch_size
    if im_w < patch_w or im_h < patch_h:
        print(f"Image size is smaller than the patch size. Skipping ...")
        return None
    patches = []
    for _ in range(n_patches):
        x, y = get_a_random_location(input_image.size, patch_size)
        patch = get_a_patch(input_image, (x, y), patch_size)
        patches.append((patch, x, y))
    
    return patches

# def get_variable_noise(sigma_min, sigma_max):
#     return sigma_min + torch.rand(1) * (sigma_max - sigma_min)

# def add_noise(xf: torch.tensor, sigma) -> torch.tensor:
#     std = torch.std(xf)
#     mu = torch.mean(xf)

#     x_centred = (xf  - mu) / std

#     x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)

#     xnoise = std * x_centred + mu

#     del std, mu, x_centred

#     return xnoise


# def tensor_2d_to_image(tensor_2d: torch.tensor) -> Image:
#     tensor_2d_np = tensor_2d.to("cpu").detach().numpy()
#     tensor_2d_np_int = (tensor_2d_np * 255.0).clip(0, 255.0).astype(np.uint8)
#     image = Image.fromarray(tensor_2d_np_int)
#     return image


# def get_noisy_image(rgb: Image, sigma) -> Image:
#     grayscale = rgb.convert("L")
    
#     im_np = np.array(grayscale)
#     im_2d = torch.tensor(im_np, dtype=torch.float32) / 255.0
#     noisy_2d = add_noise(im_2d, sigma=sigma)

#     noisy_im = tensor_2d_to_image(noisy_2d)
#     return noisy_im


def save_patches(patches: list, output_folder: str, output_file: str, 
                #  min_sigma=0.1, max_sigma=0.5,
                 sigma=0.5, 
                 overwrite=False) -> None:
    for patch, x, y in patches:
        # folder = f"{output_folder}/{output_file}-{x}-{y}"

        # patch_channels = patch.split()
        # noisy_patch_channels = []
        # for j, color in enumerate(["R", "G", "B"]):
        #     patch_single_channel = patch_channels[j]
        #     noisy_patch_single_channel = add_gaussian_noise(patch_single_channel, min_sigma, max_sigma)
        #     noisy_patch_channels.append(noisy_patch_single_channel)

        # noisy_patch = Image.merge("RGB", noisy_patch_channels)
        
        # noisy_patch = add_gaussian_noise(patch, min_sigma, max_sigma)
        
        clean_file = f"{output_folder}/clean-{x}-{y}.png"
        if not overwrite and os.path.exists(clean_file):
            continue
        patch.save(clean_file)
        # noisy_patch.save(f"{folder}/noisy.png")


def get_files_with_condition(outer_path: str, condition) -> list:
    folders = os.listdir(outer_path)
    return_files = []
    for folder in folders:
        files = os.listdir(f"{outer_path}/{folder}")
        for file in files:
            if condition(file):
                image_file = f"{outer_path}/{folder}/{file}"
                return_files.append((image_file, folder))
    return return_files


def get_sidd_files(outer_path: str, type: str, end: str = "PNG") -> list:
    condition = lambda file: file.endswith(end) and type in file
    return get_files_with_condition(outer_path, condition)


def get_and_save_patches_for_sigma(
        input_path: str, output_path: str, type="GT",  # type="GT" or "NOISY
        n_patches=32, patch_size=(256, 256), sigma=0.5) -> None:
    
    os.makedirs(output_path, exist_ok=True) # Create output folder if it doesn't exist

    sidd_files = get_sidd_files(input_path, type)
    for file, folder in tqdm(sidd_files):
        image = Image.open(file)
        # image = image.convert("L") # Convert to grayscale
        patches = get_random_patches(image, n_patches, patch_size)
        output_file = file.split("/")[-1].split(".")[0]
        file_output_path = f"{output_path}/{folder}"
        os.makedirs(file_output_path, exist_ok=True)
        save_patches(patches, file_output_path, output_file, sigma)
        # break


def get_and_save_patches(
        input_path: str, output_path: str, type="GT",  # type="GT" or "NOISY
        n_patches=32, patch_size=(256, 256), 
        # min_sigma=0.1, max_sigma=0.5,
        sigma=0.5,
        rand_seed=97) -> None:
    
    if rand_seed is not None:
        random.seed(rand_seed) # Random seed for reproducibility

    # sigma = get_variable_noise(min_sigma, max_sigma)

    get_and_save_patches_for_sigma(input_path, output_path, type, n_patches, patch_size, sigma)
