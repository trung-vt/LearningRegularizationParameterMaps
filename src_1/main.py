import os
import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in '{os.path.basename(__file__)}' ...")
import_torch_start_time = time.time() 
import torch
import_torch_end_time = time.time()
print(f"Importing torch in '{os.path.basename(__file__)}' took {import_torch_end_time - import_torch_start_time} seconds")

import datetime
from tqdm import tqdm # optional, progress bar
import wandb # optional, logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from config import get_config
from metrics import PSNR, SSIM
from unet import UNet
from pdhg import DynamicImageStaticPrimalDualNN
from data import get_dataloaders

# Optional: Use wandb to log the training process
# !wandb login
def init_wandb(config):
    project_name = config["project"]
    os.environ['WANDB_NOTEBOOK_NAME'] = project_name
    # os.environ['WANDB_MODE'] = "dryrun"  # "offline" # https://docs.wandb.ai/quickstart
    wandb.login()
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config=config,
    )


# Code taken from https://www.github.com/koflera/LearningRegularizationParameterMaps

def train_iteration(optimizer, model, loss_func, sample):
    optimizer.zero_grad(set_to_none=True)  # Zero your gradients for every batch!
    noisy_image, clean_image = sample
    # print(f"noisy_image size: {noisy_image.size()}")
    # print(f"clean_image size: {clean_image.size()}")
    denoised_image = model(noisy_image)
    loss = loss_func(denoised_image, clean_image)
    loss.backward()
    
    if loss.item() != loss.item():
        raise ValueError("NaN returned by loss function...")

    optimizer.step()

    denoised_image = denoised_image.squeeze(0).squeeze(0).squeeze(-1)
    clean_image = clean_image.squeeze(0).squeeze(0).squeeze(-1)

    psnr = PSNR(denoised_image, clean_image)
    ssim = SSIM(denoised_image, clean_image)

    return loss.item(), psnr, ssim

def train_epoch(model, data_loader, optimizer, loss_func) -> float:
    """Perform the training of one epoch.

    Parameters
    ----------
    model
        Model to be trained
    data_loader
        Dataloader with training data
    optimizer
        Pytorch optimizer, e.g. Adam
    loss_func
        Loss function to be calculated, e.g. MSE

    Returns
    -------
        training loss

    Raises
    ------
    ValueError
        loss is NaN
    """
    running_loss = 0.
    running_psnr = 0.
    running_ssim = 0.
    num_batches = len(data_loader)
    for sample in tqdm(data_loader): # tqdm helps show a nice progress bar
        loss, psnr, ssim = train_iteration(optimizer, model, loss_func, sample)
        running_loss += loss
        running_psnr += psnr
        running_ssim += ssim
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches
    return avg_loss, avg_psnr, avg_ssim



def validate_iteration(model, loss_func, sample):
    noisy_image, clean_image = sample
    denoised_image = model(noisy_image)
    loss = loss_func(denoised_image, clean_image)

    # assert len(denoised_image.size()) == 5, f"Expected 5D tensor, got {denoised_image.size()}"
    denoised_image = denoised_image.squeeze(0).squeeze(0).squeeze(-1)
    clean_image = clean_image.squeeze(0).squeeze(0).squeeze(-1)

    psnr = PSNR(denoised_image, clean_image)
    ssim = SSIM(denoised_image, clean_image)

    return loss.item(), psnr, ssim

def validate_epoch(model, data_loader, loss_func) -> float:
    """Perform the validation of one epoch.

    Parameters
    ----------
    model
        Model to be trained
    data_loader
        Dataloader with validation data
    loss_func
        Loss function to be calculated, e.g. MSE

    Returns
    -------
        validation loss
    """
    running_loss = 0.
    running_psnr = 0.
    running_ssim = 0.
    num_batches = len(data_loader)
    for sample in tqdm(data_loader): # tqdm helps show a nice progress bar
        loss, psnr, ssim = validate_iteration(model, loss_func, sample)
        running_loss += loss
        running_psnr += psnr
        running_ssim += ssim
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches
    return avg_loss, avg_psnr, avg_ssim



# Code adapted from https://www.github.com/koflera/LearningRegularizationParameterMaps

def start_training(config):
    DEVICE = config["device"]
    # Define CNN block
    unet = UNet().to(DEVICE)

    # Construct primal-dual operator with nn
    pdhg = DynamicImageStaticPrimalDualNN(
        cnn_block=unet, 
        T=config["T"],
        phase="training",
        up_bound=config["up_bound"],
        device=DEVICE,
    ).to(DEVICE)
    # pdhg.load_state_dict(torch.load("./tmp/states/2024_05_24_23_04_31.pt"))

    # TODO: Sometimes, creating the optimizer gives this error:
    #   AttributeError: partially initialized module 'torch._dynamo' has no attribute 'trace_rules' (most likely due to a circular import)
    optimizer = torch.optim.Adam(pdhg.parameters(), lr=config["learning_rate"])
    loss_function = torch.nn.MSELoss()

    num_epochs = config["epochs"]

    save_epoch = config["save_epoch"]

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    project = config["project"]
    model_name = f"model-{project}-{time}"

    # Prepare to save the model
    save_dir = config["save_dir"]
    model_states_dir = f"{save_dir}/{model_name}"
    start_epoch = 0

    # # If want to continue training from a current version, add the information here
    # model_states_dir = "./tmp/model_img_0065-scale_0_5-kernel_512-sigma_1-T_128-LeakyReLU_2024_05_26_15_55_13"
    # pdhg = torch.load(f"{model_states_dir}/model_epoch_522.pt")
    # pdhg.train(True)
    # start_epoch = 522

    os.makedirs(model_states_dir, exist_ok=True)

    # noisy_image_path = "./testcases/chest_xray_noisy.png"
    # clean_image_path = "./testcases/chest_xray_clean.png"

    # def get_image(image_path):
    #     image = Image.open(image_path)
    #     image = image.convert("L")
    #     image_data = np.asarray(image)
    #     image_data = convert_to_tensor_4D(image_data)
    #     image_data = image_data.unsqueeze(0).to(DEVICE)
    #     return image_data

    # noisy_image_data = get_image(noisy_image_path)
    # clean_image_data = get_image(clean_image_path)

    # dataset_train = MyDataset(noisy_image_path, clean_image_path)
    # dataset_valid = MyDataset(noisy_image_path, clean_image_path)

    # dataloader_train = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=1, 
    #     generator=torch.Generator(device=DEVICE),
    #     shuffle=True)
    # dataloader_valid = torch.utils.data.DataLoader(
    #     dataset_valid, batch_size=1, 
    #     generator=torch.Generator(device=DEVICE),
    #     shuffle=False)

    dataloader_train, dataloader_valid, dataloader_test = get_dataloaders(config)

    init_wandb(config)

    for epoch in range(start_epoch, num_epochs):

        # Model training
        pdhg.train(True)
        training_loss, training_psnr, training_ssim = train_epoch(pdhg, dataloader_train, optimizer, loss_function)
        # training_loss, training_psnr, training_ssim = train_iteration(optimizer, pdhg, loss_function, sample=(noisy_image_data, clean_image_data))
        pdhg.train(False)
        # print(f"Epoch {epoch+1} - TRAINING LOSS: {training_loss} - TRAINING PSNR: {training_psnr} - TRAINING SSIM: {training_ssim}")

        # Optional: Use wandb to log training progress
        wandb.log({"training_loss": training_loss})
        wandb.log({"training PSNR": training_psnr})
        wandb.log({"training SSIM": training_ssim})

        if (epoch+1) % save_epoch == 0:
            current_model_name = f"model_epoch_{epoch+1}"
            torch.save(pdhg, f"{model_states_dir}/{current_model_name}.pt")
            # print(f"\tEpoch: {epoch+1}")
            with torch.no_grad():
                torch.cuda.empty_cache()

                # Model validation
                validation_loss, validation_psnr, validation_ssim = validate_epoch(pdhg, dataloader_valid, loss_function)
                # validation_loss, validation_psnr, validation_ssim = validate_iteration(pdhg, loss_function, sample=(noisy_image_data, clean_image_data))
                print(f"Epoch {epoch+1} - VALIDATION LOSS: {validation_loss} - VALIDATION PSNR: {validation_psnr} - VALIDATION SSIM: {validation_ssim}")
                time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

                # Optional: Use wandb to log training progress
                wandb.log({"validation_loss": validation_loss})
                wandb.log({"validation PSNR": validation_psnr})
                wandb.log({"validation SSIM": validation_ssim})
                # wandb.log_model(f"{model_states_dir}/{current_model_name}.pt", name=f"model_epoch_{epoch+1}")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    # Optional: Use wandb to log training progress
    wandb.finish()
    # Save the entire model
    torch.save(pdhg, f"{model_states_dir}/final_model.pt")

    with torch.no_grad():
        torch.cuda.empty_cache()

    return pdhg


pdhg = start_training(get_config())