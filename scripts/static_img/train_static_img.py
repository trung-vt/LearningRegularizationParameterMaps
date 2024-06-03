# %%
import sys
import os

print("Importing PyTorch... (This may take a while)")
import torch
print("PyTorch version: ", torch.__version__)

from torch.utils.data import WeightedRandomSampler

from helper_functions import train_epoch, validate_epoch

sys.path.append("../../")
from data.static_img.dataset import StaticImageDenoisingDataset
from networks.static_img_primal_dual_nn import StaticImagePrimalDualNN
from networks.unet import UNet


TRAINING = [2, 4, 5, 9, 10]
VALIDATION = [11, 13]

# TODO: Try add alternatives like MPS and CPU
DEVICE_NAME = "cuda:0"
DEVICE = torch.device(DEVICE_NAME)
# DEVICE = device(DEVICE_NAME)

# %%
# Static Image Denoising Dataset [TODO: Choose a dataset]

# Make sure that the dataset was downloaded successfully
# The script to download the ... dataset can be found here: /data/static_img/download_..._data.py
# [TODO: Do we also use scaling factors?] The data samples can be created with different scaling factors.
# [TODO: Do we have to extract data?] Make sure to set extract_data to True when loading the dataset for the first time to create the static images???.
# Once the data for a specific scaling factor has been created the flag can be set to False.
dataset_train = StaticImageDenoisingDataset(
    data_path="../../data/static_img/tmp/SIDD_Small_sRGB_Only/Data",
    # ids=TRAINING,     # paper
    ids=["0064"],            # testing
    scale_factor=0.25,
    sigma=[0.1, 0.3],
    strides=[192, 192, 16],
    patches_size=[192, 192, 32],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    force_recreate_data=True,
    # force_recreate_data=False,
    device=DEVICE
)

# Create training dataloader
sampler = WeightedRandomSampler(dataset_train.samples_weights, len(dataset_train.samples_weights))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, sampler=sampler)

# Validation dataset (see note above)
dataset_valid = StaticImageDenoisingDataset(
    data_path="../../data/static_img/tmp/SIDD_Small_sRGB_Only/Data",
    # ids=VALIDATION,   # paper
    ids=["0065"],           # testing
    scale_factor=0.25,   # Make the width and height of the images half of the original size???
    sigma=[0.1, 0.3],   # Variance range in Gaussian noise added to the data???
    strides=[192, 192, 16],     # Non-overlapping patches?
    patches_size=[192, 192, 32],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    force_recreate_data=True,
    # force_recreate_data=False,
    device=DEVICE
)

# Create validation dataloader 
sampler = WeightedRandomSampler(dataset_valid.samples_weights, len(dataset_valid.samples_weights))
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, sampler=sampler)


# %%
# Define CNN block and PDHG-method
# unet = UNet(dim=3, n_ch_in=1).to(DEVICE)  # TODO: Does this 3 include the time dimension?
unet = UNet(dim=2, n_ch_in=1).to(DEVICE) # TODO: Will this limit to 2 spatial dimensions?

# Constrct primal-dual operator with nn
pdhg = StaticImagePrimalDualNN(
    unet_cnn_block=unet, 
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

    torch.cuda.empty_cache() # TODO: For performance purposes?

# Save the entire model
torch.save(pdhg, f"./tmp/model.pt")

# %%
