import unittest
import numpy as np
import torch
from data.transform import crop_and_resize, convert_to_grayscale, convert_to_numpy, convert_to_tensor_4D, get_variable_noise, add_noise
from PIL import Image

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
        sigma_min = 0
        sigma_max = 0.5
        torch.manual_seed(manual_seed)
        sigma = get_variable_noise(sigma_min, sigma_max)
        sigma_np = sigma.numpy()[0]
        print(f"sigma_np = {sigma_np}")
        assert np.allclose(sigma_np, 0.4411346), f"Expected sigma = 0.4411346, but got {sigma_np}"
        img_noisy = add_noise(img, sigma, manual_seed)
        img_noisy *= 255
        img_noisy = img_noisy.squeeze().numpy().astype(np.uint8)
        print(f"img_noisy = \n{img_noisy}")
        assert np.allclose(img_noisy, np.array([
            [ 1,  2,  3],
            [ 4,  3,  5],
            [ 9,  7,  9],
        ])), f"Expected img_noisy = [[ 1,  2,  3], [ 4, 2, 5], [12, 6, 10]], but got {img_noisy}"