import torch
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from PIL import Image

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from lib import pdhg
from lib import pdhg

def convert_to_grayscale(image: Image) -> Image:
    return image.convert('L')

def resize_image(image, scale_factor):
    Nx_,Ny_ = int(np.floor(scale_factor * image.width )), int(np.floor(scale_factor * image.height ))
    image = image.resize( (Nx_, Ny_) )
    return image

def convert_to_numpy(image):
    image_data = np.asarray(image)
    return image_data

def convert_to_tensor(image_data):
    xf = []
    xf.append(image_data)
    xf = np.stack(xf, axis=-1)
    xf = torch.tensor(xf, dtype=torch.float)
    xf = xf.unsqueeze(0) / 255
    return xf

def get_variable_noise(sigma_min, sigma_max):
    return sigma_min + torch.rand(1) * (sigma_max - sigma_min)

def add_noise(xf: torch.tensor, sigma) -> torch.tensor:
    std = torch.std(xf)
    mu = torch.mean(xf)

    x_centred = (xf  - mu) / std

    x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)

    xnoise = std * x_centred + mu

    return xnoise

def PSNR(original, compressed): 
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0): # MSE is zero means no noise is present in the signal. 
                  # Therefore PSNR have no importance. 
        return 100
    # max_pixel = 255.0
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse)) 
    return psnr

def SSIM(original, compressed):
    original = original.squeeze(0)
    original = original.squeeze(-1)
    original = original.detach().to("cpu").numpy()
    compressed = compressed.squeeze(0)
    compressed = compressed.squeeze(-1)
    compressed = compressed.detach().to("cpu").numpy()
    # print(f"original: {original.shape}")
    # print(f"compressed: {compressed.shape}")
    return structural_similarity(original, compressed, data_range=1)


def test_reconstruct_with_PDHG():
    grayscale_image = Image.open("turtle clean.png")
    grayscale_image = resize_image(grayscale_image, 0.125)
    image_data = convert_to_numpy(grayscale_image)

    print(f"Image data shape: {image_data.shape}")

    image_data = convert_to_tensor(image_data)
    print(f"Image size: {image_data.size()}")
    print(f"PSNR of original image: {PSNR(image_data, image_data)} dB")
    plt.imshow(image_data.squeeze(0).to("cpu"), cmap='gray')
    plt.show();

    TEST_SIGMA = 0.5  # Relatively high noise
    constant_noise_img = add_noise(image_data, sigma=TEST_SIGMA)
    print(f"PSNR of constant noise image: {PSNR(image_data, constant_noise_img):.2f} dB")
    print(f"SSIM of constant noise image: {SSIM(image_data, constant_noise_img):.2f}")
    plt.imshow(constant_noise_img.squeeze(0).to("cpu"), cmap='gray')
    plt.show();

    # TEST_LAMBDA = 0.1 # A test value to see the effect of lambda on regularization
    # TEST_LAMBDA = 0.2 # A test value to see the effect of lambda on regularization
    TEST_LAMBDA = 0.04
    pdhg_input = constant_noise_img.unsqueeze(0)
    print(f"PDHG input size: {pdhg_input.size()}")
    reconstructed_img = pdhg.reconstruct(
        pdhg_input, 
        lambda_reg=TEST_LAMBDA, 
        T=128
        # T=1000
        )
    # Clip the values to 0 and 1
    reconstructed_img = torch.clamp(reconstructed_img, 0, 1)
    resconstructed_image_data = reconstructed_img
    print(f"Reconstructed image data shape: {resconstructed_image_data.shape}")
    print(f"PSNR of reconstructed image: {PSNR(image_data, resconstructed_image_data):.2f} dB")
    plt.imshow(resconstructed_image_data.squeeze(0).squeeze(0).to("cpu").detach().numpy(), cmap='gray')
    plt.show();

    with torch.no_grad():
        torch.cuda.empty_cache()

    print("""
In this example, a lot of noise has been applied to the original image. The PDHG algorithm tries to reconstruct the image from the noisy image. It did remove some noise and improved the PSNR value. However, the quality has been degraded significantly. We will see whether we can improve this by learning a set of parameters map.
""")
    
    # The lambda parameter is the regularization parameter. The higher the lambda, the more the regularization. The T parameter is the number of iterations. The higher the T, the more the iterations. The PSNR value is the Peak Signal to Noise Ratio. The higher the PSNR, the better the reconstruction.

test_reconstruct_with_PDHG()