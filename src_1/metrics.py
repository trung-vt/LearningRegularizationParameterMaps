import os
import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in '{os.path.basename(__file__)}' ...")
import_torch_start_time = time.time() 
import torch
import_torch_end_time = time.time()
print(f"Importing torch in '{os.path.basename(__file__)}' took {import_torch_end_time - import_torch_start_time} seconds")

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def PSNR(original, compressed): 
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0): # MSE is zero means no noise is present in the signal. 
                  # Therefore PSNR have no importance. 
        return 100
    # max_pixel = 255.0
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse)) 
    return psnr


def SSIM(tensor_2D_a: torch.Tensor, tensor_2D_b: torch.Tensor, data_range: float=1) -> float:
    return structural_similarity(
        tensor_2D_a.to("cpu").detach().numpy(), 
        tensor_2D_b.to("cpu").detach().numpy(),
        data_range=data_range)