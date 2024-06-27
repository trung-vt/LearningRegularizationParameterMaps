import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np
import pandas as pd

def PSNR(original, compressed): 
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0): # MSE is zero means no noise is present in the signal. 
                  # Therefore PSNR have no importance. 
        return 100
    # max_pixel = 255.0
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse)) 

    del mse

    return psnr


def SSIM(tensor_2D_a: torch.Tensor, tensor_2D_b: torch.Tensor, data_range: float=1) -> float:
    return structural_similarity(
        tensor_2D_a.to("cpu").detach().numpy(), 
        tensor_2D_b.to("cpu").detach().numpy(),
        data_range=data_range)
    
    
def compare(noisy_4d, clean_4d):
    clean_np = clean_4d.squeeze(-1).squeeze(0).detach().cpu().numpy()
    noisy_np = noisy_4d.squeeze(-1).squeeze(0).detach().cpu().numpy()
    
    mse_val = ((clean_np - noisy_np) ** 2).mean()
    psnr_val = peak_signal_noise_ratio(clean_np, noisy_np, data_range=1)
    ssim_val = structural_similarity(clean_np, noisy_np, data_range=1)
    
    df_results = pd.DataFrame({
        "MSE": [mse_val],
        "PSNR": [psnr_val],
        "SSIM": [ssim_val]
    })
    
    return df_results