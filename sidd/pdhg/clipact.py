# Code taken from https://www.github.com/koflera/LearningRegularizationParameterMaps


import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in {__file__} ...")
import_torch_start_time = time.time() 
import torch
print(f"Importing torch took {time.time() - import_torch_start_time} seconds")

from torch import nn


class ClipAct(nn.Module):
    def forward(self, x, threshold):
        return clipact(x, threshold)


def clipact(x, threshold):
    is_complex = x.is_complex()
    if is_complex:
        x = torch.view_as_real(x)
        threshold = threshold.unsqueeze(-1)
    x = torch.clamp(x, -threshold, threshold)
    if is_complex:
        x = torch.view_as_complex(x)
    return x