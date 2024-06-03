# %%
import sys
import os

import torch
from torch.utils.data import WeightedRandomSampler

from helper_functions import train_epoch, validate_epoch

sys.path.append("../../")
from data.dyn_img_static.dataset import DynamicImageStaticDenoisingDataset
from networks.dyn_img_static_primal_dual_nn import DynamicImageStaticPrimalDualNN
from networks.unet_2d import UNet


TRAINING = [2, 4, 5, 9, 10]
VALIDATION = [11, 13]

DEVICE = torch.device("cuda:0")

# %%
# Dynamic Image Denoising Dataset

data_path = "../../data/dyn_img_static/tmp/SIDD_Small_sRGB_Only/Data"
# Make sure that the dataset was downloaded successfully
# The script to download the MOT17Det dataset can be found here: /data/dyn_img/download_mot_data.py
# The data samples can be created with different scaling factors.
# Make sure to set extract_data to True when loading the dataset for the first time to create the dynamic images.
# Once the data for a specific scaling factor has been created the flag can be set to False.
dataset_train = DynamicImageStaticDenoisingDataset(
    data_path=data_path,
    # ids=TRAINING,     # paper
    ids=["0001"],            # testing
    scale_factor=1,
    sigma=[0.1, 0.3],
    strides=[192, 192, 1],
    patches_size=[192, 192, 1],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    extract_data=True,
    # extract_data=False,
    device=DEVICE
)

# Create training dataloader
sampler = WeightedRandomSampler(dataset_train.samples_weights, len(dataset_train.samples_weights))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, sampler=sampler)

# Validation dataset (see note above)
dataset_valid = DynamicImageStaticDenoisingDataset(
    data_path=data_path,
    # ids=VALIDATION,   # paper
    ids=["0002"],           # testing
    scale_factor=1,
    sigma=[0.1, 0.3],
    strides=[192, 192, 1],
    patches_size=[192, 192, 1],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    extract_data=True,
    # extract_data=False,
    device=DEVICE
)

# Create validation dataloader 
sampler = WeightedRandomSampler(dataset_valid.samples_weights, len(dataset_valid.samples_weights))
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, sampler=sampler)


# %%
# Define CNN block and PDHG-method
unet = UNet(dim=3, n_ch_in=1).to(DEVICE)

# Constrct primal-dual operator with nn
pdhg = DynamicImageStaticPrimalDualNN(
    cnn_block=unet, 
    T=128,
    phase="training",
    up_bound=0.5,
    # Select mode:
    mode="lambda_cnn",
    # mode="lambda_xy_t",
    # mode="lambda_xyt",
).to(DEVICE)

optimizer = torch.optim.Adam(pdhg.parameters(), lr=1e-4)
loss_function = torch.nn.MSELoss()

num_epochs = 2          # testing
# num_epochs = 100      # paper

model_states_dir = "./tmp/states"
os.makedirs(model_states_dir, exist_ok=True)

for epoch in range(num_epochs):

    # Model training
    pdhg.train(True)
    training_loss = train_epoch(pdhg, dataloader_train, optimizer, loss_function)
    pdhg.train(False)
    print("TRAINING LOSS: ", training_loss)

    if (epoch+1) % 2 == 0:

        with torch.no_grad():

            # Model validation
            validation_loss = validate_epoch(pdhg, dataloader_valid, loss_function)
            print("VALIDATION LOSS: ", validation_loss)
            torch.save(pdhg.state_dict(), f"{model_states_dir}/epoch_{str(epoch).zfill(3)}.pt")

    torch.cuda.empty_cache()

# Use pdhg for prediction
clean_image = 
with torch.no_grad():
    for i, data in enumerate(dataloader_train):
        # xf = data["xf"].to(DEVICE)
        # y = data["y"].to(DEVICE)
        # xf_hat = pdhg(xf)

        # print(f"xf shape: {xf.shape}")
        # print(f"y shape: {y.shape}")
        # print(f"xf_hat shape: {xf_hat.shape}")

        print(f"type(data): {type(data)}")
        print(f"len(data): {len(data)}")
        print(f"type(data[0]): {type(data[0])}")
        output = pdhg(data[0])
        print(f"type(output): {type(output)}")

        # Save output as image
        output = output.squeeze().cpu().numpy()
        print(f"output shape: {output.shape}")
        print(f"output type: {output.dtype}")

        # Save to image
        import numpy as np
        from PIL import Image
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        output = Image.fromarray(output)
        output.save(f"./tmp/images/output_{i}.png")

        # break

# Save the entire model
torch.save(pdhg, f"./tmp/model.pt")

# %%
