import torch
import os
import datetime
import json
import yaml
from tqdm import tqdm
import wandb

from data.get_data_loaders import get_data_loaders
from networks.unet_3d import UNet3d
from networks.static_img_primal_dual_nn import StaticImagePrimalDualNN
from metrics.metrics import PSNR, SSIM

print(f"Using new version 2 of train.py")

# Code adapted from https://www.github.com/koflera/LearningRegularizationParameterMaps

def common_iter(sample, model, loss_func, stage:str):
    noisy_5d, clean_5d = sample
    denoised_5d = model(noisy_5d)
    loss = loss_func(denoised_5d, clean_5d)

    denoised_2d = denoised_5d.squeeze(0).squeeze(0).squeeze(-1)
    clean_2d = clean_5d.squeeze(0).squeeze(0).squeeze(-1)
    psnr = PSNR(denoised_2d, clean_2d)
    ssim = SSIM(denoised_2d, clean_2d)
    
    wandb.log({f"{stage}_iter_loss": loss.item()})
    wandb.log({f"{stage}_iter_PSNR": psnr})
    wandb.log({f"{stage}_iter_SSIM": ssim})

    # Free up memory
    del denoised_5d # Delete output of model
    del denoised_2d # Delete auxiliary variable
    del clean_2d # Delete auxiliary variable
    del noisy_5d # Noisy image is generated each time so can delete it
    del clean_5d # TODO: Is this a copy that I can delete or a reference to the original?
    
    return loss, psnr, ssim


def train_iter(sample, model, loss_func, optimizer):
    optimizer.zero_grad(set_to_none=True)  # Zero your gradients for every batch! TODO: Why?
    
    loss, psnr, ssim = common_iter(
        sample=sample, model=model, loss_func=loss_func, stage="training"
    )
    
    loss.backward()
    if loss.item() != loss.item():
        raise ValueError("NaN returned by loss function...")
    optimizer.step()
    
    loss_value = loss.item()
    del loss # Free up memory
    
    return loss_value, psnr, ssim



def validate_iter(sample, model, loss_func, optimizer=None):
    loss, psnr, ssim = common_iter(
        sample=sample, model=model, loss_func=loss_func, stage="val"
    )

    loss_value = loss.item()
    del loss # Free up memory
    
    return loss_value, psnr, ssim


def perform_epoch(data_loader, model, loss_func, optimizer, perform_iter):
    running_loss = 0.
    running_psnr = 0.
    running_ssim = 0.
    num_batches = len(data_loader)
    # for sample in tqdm(data_loader): # tqdm helps show a nice progress bar
    for sample in data_loader:
        loss_value, psnr, ssim = perform_iter(
            sample=sample, model=model, loss_func=loss_func, optimizer=optimizer
        )
        running_loss += loss_value
        running_psnr += psnr
        running_ssim += ssim

    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches

    del running_loss, running_psnr, running_ssim, num_batches # Explicitly delete variables to free up memory

    return avg_loss, avg_psnr, avg_ssim


# Optional: Use wandb to log the training process
# !wandb login
def init_wandb(config):
    project_name = config["project"]
    os.environ['WANDB_NOTEBOOK_NAME'] = project_name
    os.environ['WANDB_MODE'] = config["wandb_mode"] # https://docs.wandb.ai/quickstart
    wandb.login()
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config=config,
    )


def start_training(config, get_datasets, pretrained_model_path=None, is_state_dict=False, start_epoch=0):
    
    data_loader_train, data_loader_valid, data_loader_test = get_data_loaders(config, get_datasets(config))

    del data_loader_test # Not used for now

    if pretrained_model_path is None or is_state_dict:
        # Define CNN block
        unet = UNet3d(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            init_filters=config["init_filters"],
            n_blocks=config["n_blocks"],
            activation=config["activation"],
            downsampling_kernel=config["downsampling_kernel"],
            downsampling_mode=config["downsampling_mode"],
            upsampling_kernel=config["upsampling_kernel"],
            upsampling_mode=config["upsampling_mode"],
        ).to(config["device"])

        # Construct primal-dual operator with nn
        pdhg = StaticImagePrimalDualNN(
            cnn_block=unet, 
            T=config["T"],
            phase="training",
            up_bound=config["up_bound"],
        ).to(config["device"])
        if is_state_dict:
            pdhg.load_state_dict(torch.load(f"{model_states_dir}/{pretrained_model_path}.pt"))
    else:
        pdhg = torch.load(f"{model_states_dir}/{pretrained_model_path}.pt")

    pdhg.train(True)

    # TODO: Sometimes, creating the optimizer gives this error:
    #   AttributeError: partially initialized module 'torch._dynamo' has no attribute 'trace_rules' (most likely due to a circular import)
    optimizer = torch.optim.Adam(pdhg.parameters(), lr=config["learning_rate"])
    loss_function = torch.nn.MSELoss()

    num_epochs = config["epochs"]

    save_epoch_local = config["save_epoch_local"]
    save_epoch_wandb = config["save_epoch_wandb"]

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    project = config["project"]
    model_name = f"model-{project}-{time}"

    # Prepare to save the model
    save_dir = config["save_dir"]
    model_states_dir = f"{save_dir}/{model_name}"

    os.makedirs(model_states_dir, exist_ok=True)

    def log_to_files():
        with open(f"{model_states_dir}/config.json", "w") as f:
            json.dump(config, f, indent=4)
        with open(f"{model_states_dir}/config.yaml", "w") as f:
            yaml.dump(config, f)
        with open(f"{model_states_dir}/config.txt", "w") as f:
            f.write(str(config))
        with open(f"{model_states_dir}/unet.txt", "w") as f:
            f.write(str(unet))
        with open(f"{model_states_dir}/pdhg_net.txt", "w") as f:
            f.write(str(pdhg))

        def log_data(data_loader, stage):
            dataset = data_loader.dataset
            with open(f"{model_states_dir}/data_loader_{stage}.txt", "w") as f:
                f.write(f"Batch size: {data_loader.batch_size}\n\n")
                f.write(f"Number of batches: {len(data_loader)}\n\n")
                f.write(f"Number of samples: {len(dataset)}\n\n")
                # f.write(f"Samples weights:\n{str(dataset.samples_weights)}\n\n")
                f.write(f"Sample 0 size:\n{str(len(dataset[0]))}  {str(dataset[0][0].size())}\n\n")
                f.write(f"Sample 0:\n{str(dataset[0])}\n\n")
        log_data(data_loader_train, "train")
        log_data(data_loader_valid, "val")
        # log_data(data_loader_test, "test")

    log_to_files()

    init_wandb(config)

    # for epoch in range(start_epoch, num_epochs):
    for epoch in tqdm(range(start_epoch, num_epochs)):
        wandb.log({"epoch": epoch+1})
        # Model training
        pdhg.train(True)
        train_loss, train_psnr, train_ssim = perform_epoch(
            data_loader_train, pdhg, loss_function, optimizer, train_iter
        )
        # Optional: Use wandb to log training progress
        wandb.log({"train_loss": train_loss, "train_PSNR": train_psnr, "train_SSIM": train_ssim})

        del train_loss, train_psnr, train_ssim

        pdhg.train(False)
        with torch.no_grad():
            torch.cuda.empty_cache()

            # Model validation
            val_loss, val_psnr, val_ssim = perform_epoch(
                data_loader_valid, pdhg, loss_function, optimizer, validate_iter
            )
            # Optional: Use wandb to log training progress
            wandb.log({"val_loss": val_loss, "val_PSNR": val_psnr, "val_SSIM": val_ssim})

            torch.cuda.empty_cache()


        if (epoch+1) % save_epoch_local == 0:
            current_model_name = f"model_epoch_{epoch+1}"
            torch.save(pdhg, f"{model_states_dir}/{current_model_name}.pt")
            
            print(f"Epoch {epoch+1} - VALIDATION LOSS: {val_loss} - VALIDATION PSNR: {val_psnr} - VALIDATION SSIM: {val_ssim}")

        if (epoch+1) % save_epoch_wandb == 0:
            wandb.log_model(f"{model_states_dir}/{current_model_name}.pt", name=f"model_epoch_{epoch+1}")
            
        del val_loss, val_psnr, val_ssim

        torch.cuda.empty_cache()


    # Save the entire model
    torch.save(pdhg, f"{model_states_dir}/final_model.pt")
    
    wandb.log_model(f"{model_states_dir}/final_model.pt", name=f"final_model")
    wandb.finish()
    
    with torch.no_grad():
        torch.cuda.empty_cache()

    return pdhg