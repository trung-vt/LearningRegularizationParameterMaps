import numpy as np
from PIL import Image
import torch

from metrics import SSIM

def test_SSIM():
    rgb_image = Image.open("turtle clean.png")

    # grayscale_image = convert_to_grayscale(rgb_image)
    # Convert to grayscale
    grayscale_image = rgb_image.convert("L")
    # image_data = convert_to_numpy(grayscale_image)
    # Convert to numpy array
    image_data = np.array(grayscale_image)
    # image_data = convert_to_tensor(image_data)
    # Convert to tensor
    image_data = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    print(f"SSIM of original image: {SSIM(image_data, image_data)}")
    plt.imshow(image_data.squeeze(0).to("cpu"), cmap='gray')
    plt.show();

    noisy_img = add_noise(image_data, sigma=0.5)
    print(f"SSIM of constant noise image: {SSIM(image_data, noisy_img):.2f}")
    plt.imshow(noisy_img.squeeze(0).to("cpu"), cmap='gray')
    plt.show();

with torch.no_grad():
    torch.cuda.empty_cache()

test_SSIM()