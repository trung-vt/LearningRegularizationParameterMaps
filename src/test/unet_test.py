# %%
import torch

import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet

def test_unet():
    # Example usage
    model = UNet()
    # input_tensor = torch.randn(1, 1, 64, 64, 64)  # batch size of 1, 1 channel, 64x64x64 volume
    input_tensor = torch.randn(1, 1, 512, 512, 1)  # batch size of 1, 1 channel, 64x64x64 volume
    output = model(input_tensor)
    # print(output.shape)  # should be (1, 2, 64, 64, 64)
    print(output.shape)  # should be (1, 2, 512, 512, 1)

    print(f"\n{model}")

    with torch.no_grad():
        torch.cuda.empty_cache()

    # Delete the model and the output tensor
    del model
    del output
    torch.cuda.empty_cache()

test_unet()
with torch.no_grad():
    torch.cuda.empty_cache()