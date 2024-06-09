import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in {__file__} ...")
import_torch_start_time = time.time() 
import torch
print(f"Importing torch took {time.time() - import_torch_start_time} seconds")


def numpy_image_256_to_tensor_with_channel_and_time(image_numpy):
    xf = torch.tensor(image_numpy, dtype=torch.float)
    xf = xf.unsqueeze(0)
    xf = xf.unsqueeze(-1)
    xf = xf / 255 # Normalise from [0, 255] to [0, 1]
    return xf

def tensor_with_channel_and_time_to_numpy_image_256(image_tensor):
    image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.squeeze(-1)
    image_tensor = image_tensor * 255 # Denormalise from [0, 1] to [0, 255]
    return image_tensor.to("cpu").detach().numpy().astype('uint8')
    