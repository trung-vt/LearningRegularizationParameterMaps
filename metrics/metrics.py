import torch
from skimage.metrics import structural_similarity

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