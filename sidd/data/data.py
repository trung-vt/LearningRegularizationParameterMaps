import os
import random
from PIL import Image
from tqdm import tqdm # progress bar

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


def save_patches(patches: list, output_folder: str, output_file: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for i, patch in enumerate(patches):
        patch.save(f"{output_folder}/{output_file}_{i}.PNG")


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
        n_patches=32, patch_size=(256, 256),
        rand_seed=97) -> None:
    
    os.makedirs(output_path, exist_ok=True) # Create output folder if it doesn't exist

    if rand_seed is not None:
        random.seed(rand_seed) # Random seed for reproducibility

    sidd_files = get_sidd_files(input_path, type)
    for file in tqdm(sidd_files):
        image = Image.open(file)
        patches = get_random_patches(image, n_patches, patch_size)
        output_file = file.split("/")[-1].split(".")[0]
        save_patches(patches, output_path, output_file)
