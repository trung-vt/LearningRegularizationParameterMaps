import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in {__file__} ...")
import_torch_start_time = time.time()
import torch
print(f"Importing torch took {time.time() - import_torch_start_time} seconds")


def get_variable_noise(sigma_min, sigma_max):
    return sigma_min + torch.rand(1) * (sigma_max - sigma_min)

def add_gaussian_noise(xf: torch.tensor, sigma) -> torch.tensor:
    std = torch.std(xf)
    mu = torch.mean(xf)

    x_centred = (xf  - mu) / std

    x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)

    xnoise = std * x_centred + mu

    del std, mu, x_centred

    return xnoise