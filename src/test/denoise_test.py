import torch
import numpy as np
import hydra
import pprint
from PIL import Image
import os

def convert_to_tensor(image_data):
    xf = []
    xf.append(image_data)
    xf = np.stack(xf, axis=-1)
    xf = torch.tensor(xf, dtype=torch.float)
    xf = xf.unsqueeze(0) / 255
    return xf

def test_denoise(config, showing=False):
    """
    Testing denoising with pre-trained parameters.
    """
    clean_image = Image.open(config.data.val_path.clean)
    clean_image = convert_to_tensor(clean_image)

    noisy_image = Image.open(config.data.val_path.noisy)
    noisy_image = convert_to_tensor(noisy_image)

    # Save the results
    # folder_name = f"./tmp/images/presentation-img_{img_id}-scale_{str(scale).replace('.', '_')}-sigma_{str(sigma).replace('.', '_')}-best_lambda_{str(best_lambda).replace('.', '_')}-kernel_{k_w}-model_{model_name}-activation_{activation}-trained_on_{trained_on}-time_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    folder_name = "./tmp"
    os.makedirs(folder_name, exist_ok=True)

    # Denoise the image using a single lambda
    best_lambda = 0.07  # From the slides?
    x_denoised_single_lambda = reconstruct_with_PDHG(
        noisy_image.unsqueeze(0), best_lambda, 
        T=128
        # T=1024
    )

    pdhg = torch.load(config.outputs.model_path)
    pdhg.eval()


    lambda_map = pdhg.get_lambda_cnn(noisy_image)

    print(f"Max regularisation value: {lambda_map.max()}")
    print(f"Min regularisation value: {lambda_map.min()}")

    lambda_map_xy_1 = lambda_map[:, 0, :, :]
    lambda_map_xy_2 = lambda_map[:, 1, :, :]
    lambda_map_t = lambda_map[:, 2, :, :]

    assert torch.allclose(lambda_map_xy_1, lambda_map_xy_2, atol=1e-3)
    
    # Combine lambda x and lambda y back to lambda map
    lambda_map_xy_double = torch.stack([lambda_map_xy_1, lambda_map_xy_2, lambda_map_t], dim=1)
    
    x_denoised_lambda_map = reconstruct_with_PDHG(
        noisy_image.unsqueeze(0), lambda_map_xy_double, 
        T=128
        # T=1024
    )


@hydra.main(config_path=".", config_name="model_turtle_config")
def main(config):

    pprint.pprint(dict(config))
    test_denoise(config, showing=True)

main()