import unittest
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from data.transform import crop_and_resize, convert_to_grayscale, convert_to_numpy, convert_to_tensor_4D, get_variable_noise, add_noise

class TestTransformMethods(unittest.TestCase):
    def test_add_noise(self):
        manual_seed = 42
        img_np = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9],
        ]).astype(np.uint8)
        img = Image.fromarray(img_np)
        img = convert_to_grayscale(img)
        img = convert_to_numpy(img)
        img = convert_to_tensor_4D(img) # divided by 255 in here
        print(f"Before adding noise: img = \n{img}\n")
        sigma_min = 0
        sigma_max = 0.5
        torch.manual_seed(manual_seed)
        sigma = get_variable_noise(sigma_min, sigma_max)
        sigma_np = sigma.numpy()[0]
        print(f"sigma_np = {sigma_np}\n")
        assert np.allclose(sigma_np, 0.4411346), f"Expected sigma = 0.4411346, but got {sigma_np}"
        img_noisy = add_noise(img, sigma, manual_seed)
        print(f"After adding noise: img_noisy = \n{img_noisy}\n")
        img_noisy *= 255
        img_noisy = torch.clamp(img_noisy, 0, 255)
        img_noisy = img_noisy.squeeze().numpy().astype(np.uint8)
        print(f"img_noisy = \n{img_noisy}\n")
        assert np.allclose(img_noisy, np.array([
            [ 1,  2,  3],
            [ 4,  3,  5],
            [ 9,  7,  9],
        ])), f"Expected img_noisy = [[ 1,  2,  3], [ 4, 2, 5], [12, 6, 10]], but got {img_noisy}"
        
    def test_add_noise_0(self):
        manual_seed = 42
        img_np = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]).astype(np.uint8)
        img = Image.fromarray(img_np)
        img = convert_to_grayscale(img)
        img = convert_to_numpy(img)
        img = convert_to_tensor_4D(img) # divided by 255 in here
        print(f"Before adding noise: img = \n{img}\n")
        sigma_min = 0
        sigma_max = 0.5
        torch.manual_seed(manual_seed)
        sigma = get_variable_noise(sigma_min, sigma_max)
        sigma_np = sigma.numpy()[0]
        print(f"sigma_np = {sigma_np}\n")
        assert np.allclose(sigma_np, 0.4411346), f"Expected sigma = 0.4411346, but got {sigma_np}"
        img_noisy = add_noise(img, sigma, manual_seed)
        print(f"After adding noise: img_noisy = \n{img_noisy}\n")
        img_noisy *= 255
        img_noisy = torch.clamp(img_noisy, 0, 255)
        img_noisy = img_noisy.squeeze().numpy().astype(np.uint8)
        print(f"img_noisy = \n{img_noisy}\n")
        assert np.allclose(img_noisy, np.array([
            [ 1,  2,  3],
            [ 4,  3,  5],
            [ 9,  7,  9],
        ])), f"Expected img_noisy = [[ 1,  2,  3], [ 4, 2, 5], [12, 6, 10]], but got {img_noisy}"
        
        
        
def test_convert_greyscale():
    from data.chest_xray_load_images import load_images_chest_xray
    CHEST_XRAY_BASE_DATA_PATH = "../data/chest_xray"
    # for img in load_images_SIDD(["0065"], False):
    for img in load_images_chest_xray(f"{CHEST_XRAY_BASE_DATA_PATH}/train/NORMAL", [0]):
        img = convert_to_grayscale(img)
        plt.imshow(img, cmap='gray') # cmap='gray' for proper display in black and white. It does not convert the image to grayscale.
        
        
def test_crop_to_square_and_resize():
    from data.chest_xray_load_images import load_images_chest_xray
    CHEST_XRAY_BASE_DATA_PATH = "../data/chest_xray"
    # for img in load_images_SIDD(["0083"], False):
    for img in load_images_chest_xray(f"{CHEST_XRAY_BASE_DATA_PATH}/train/NORMAL", [0]):
        plt.imshow(img, cmap='gray')
        plt.show();
        img = crop_and_resize(img, 120)
        print(img.size)
        plt.imshow(img, cmap='gray') # cmap='gray' for proper display in black and white. It does not convert the image to grayscale.
        plt.show();
