# NOTE: Importing torch the first time will always take a long time!
import torch
import torch.nn as nn

import torch.multiprocessing as mp
# Set the start method as soon as the script starts
mp.set_start_method('spawn', force=True)

import wandb # for logging

import os
from hydra import initialize, compose
from omegaconf import OmegaConf

from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import chest_xray_data, net_theta, full_pdhg_net

def load_config():
    # https://gist.github.com/bdsaglam/586704a98336a0cf0a65a6e7c247d248?permalink_comment_id=4478589#gistcomment-4478589
    with initialize(version_base=None, config_path="conf"):
        config = compose(config_name='config.yaml')
    return config


def get_optimizer(optimizer_name, learning_rate, model_params):
    if optimizer_name == "Adam":
        return torch.optim.Adam(model_params, lr=learning_rate)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model_params, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_criterion(loss_function):
    if loss_function == "MSE":
        return nn.MSELoss()
    elif loss_function == "L1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

# Optional: Use wandb to log the training process
def get_wandb_logger(config):

    wandb.login()

    os.environ['WANDB_NOTEBOOK_NAME'] = config.project_name
    # return wandb.init(project="chest-xray-autoencoder", entity="wandb", reinit=True)

    # start a new wandb run to track this script
    return wandb.init(
            # set the wandb project where this run will be logged
            project=config.project_name,

            # track hyperparameters and run metadata
            config={
                "architecture": config.architecture.model,
                "unet_size": config.architecture.unet_size,
                "unet": net_theta.UNet(),

                "dataset" : config.data.dataset,
                "scale_factor": config.data_loader.scale_factor,
                "patch_size": config.data_loader.patch_size,
                "sigma": config.data_loader.noise_sigma,
                "num_images": config.data_loader.num_images,

                "learning_rate": config.train_params.learning_rate,
                "optimizer": config.train_params.optimizer.name,
                "loss_function": config.train_params.criterion,
                "T": config.train_params.T,
                "up_bound": config.train_params.up_bound,
                "activation": config.train_params.activation,
                "epochs": config.train_params.epochs,
            }
        )



def evaluate(model, val_loader, criterion, device="cuda", is_multichannel=False):
    model.eval()
    running_loss = 0.0
    val_ssim = 0.0
    val_psnr = 0.0
    num_batches = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            noisy_image, clean_image = data

            # IMPORTANT: Move the data to the device before passing it to the model
            noisy_image = noisy_image.to(device)
            clean_image = clean_image.to(device)
            denoised_image = model(noisy_image)

            loss = criterion(denoised_image, clean_image)

            running_loss += loss.item()
            ssim, psnr = get_metrics(denoised_image, clean_image, is_multichannel=is_multichannel)
            val_ssim += ssim
            val_psnr += psnr

        val_loss = running_loss / num_batches
        val_ssim = val_ssim / num_batches
        val_psnr = val_psnr / num_batches
        return val_loss, val_ssim, val_psnr


def test_evaluate():
    model = nn.Conv2d(3, 3, 3, padding=1)
    val_loader = torch.utils.data.DataLoader(torch.randn(10, 3, 256, 256), batch_size=2)
    criterion = nn.MSELoss()
    evaluate(model, val_loader, criterion)


def train_one_epoch(model, train_loader, criterion, optimizer, device, is_multichannel):
    model.train()
    running_loss = 0.0
    train_ssim = 0.0
    train_psnr = 0.0
    num_batches = len(train_loader)
    for i, data in enumerate(tqdm(train_loader)):
        noisy_image, clean_image = data
        optimizer.zero_grad()

        # IMPORTANT: Move the data to the device before passing it to the model
        noisy_image = noisy_image.to(device)
        clean_image = clean_image.to(device)
        denoised_image = model(noisy_image)

        loss = criterion(denoised_image, clean_image)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ssim, psnr = get_metrics(denoised_image, clean_image, is_multichannel=is_multichannel)
        train_ssim += ssim
        train_psnr += psnr

    train_loss = running_loss / num_batches
    train_ssim = train_ssim / num_batches
    train_psnr = train_psnr / num_batches
    return train_loss, train_ssim, train_psnr
    


def test(model, test_loader, criterion, device="cuda", is_multichannel=False):
    model.eval()
    running_loss = 0.0
    test_ssim = 0.0
    test_psnr = 0.0
    num_batches = len(test_loader)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            noisy_image, clean_image = data

            # IMPORTANT: Move the data to the device before passing it to the model
            noisy_image = noisy_image.to(device)
            clean_image = clean_image.to(device)
            denoised_image = model(noisy_image)

            loss = criterion(denoised_image, clean_image)
            
            running_loss += loss.item()
            ssim, psnr = get_metrics(denoised_image, clean_image, is_multichannel)
            test_ssim += ssim
            test_psnr += psnr

        test_loss = running_loss / num_batches
        test_ssim = test_ssim / num_batches
        test_psnr = test_psnr / num_batches
        return test_loss, test_ssim, test_psnr



def get_metrics(denoised_image, clean_image, is_multichannel=False):
    print(f"denoised_image.shape: {denoised_image.shape}")
    print(f"clean_image.shape: {clean_image.shape}")

    # Expect 4 or 5 dimensions
    assert len(denoised_image.shape) in [4, 5], f"Expected 4D or 5D tensor, got {len(denoised_image.shape)}D denoised image tensor"
    assert len(clean_image.shape) in [4, 5], f"Expected 4D or 5D tensor, got {len(clean_image.shape)}D clean image tensor"

    # The dimensions are batch_size, channels, height, width (and maybe time)
    # We need to remove the batch_size and channel dimensions
    denoised_image = denoised_image.squeeze(0).squeeze(0)
    clean_image = clean_image.squeeze(0).squeeze(0)
    print(f"denoised_image.shape: {denoised_image.shape}")
    print(f"clean_image.shape: {clean_image.shape}")

    # If we have time dimension, remove it too
    if len(denoised_image.shape) == 3:
        denoised_image = denoised_image.squeeze(-1)
        clean_image = clean_image.squeeze(-1)
    print(f"denoised_image.shape: {denoised_image.shape}")
    print(f"clean_image.shape: {clean_image.shape}")

    assert denoised_image.shape == clean_image.shape, f"Shape mismatch: denoised_image_shape={denoised_image.shape} does not match clean_image_shape={clean_image.shape}"

    print(f"denoised_image.shape: {denoised_image.shape}")
    print(f"clean_image.shape: {clean_image.shape}")
    assert len(denoised_image.shape) == 2, f"Expected 2D image, got {len(denoised_image.shape)}D denoised image"
    assert len(clean_image.shape) == 2, f"Expected 2D image, got {len(clean_image.shape)}D clean image"
    
    # Clip the denoised image to [0, 1]
    denoised_image = torch.clamp(denoised_image, 0, 1)
    # clean_image = torch.clamp(clean_image, 0, 1)

    # To CPU
    denoised_image = denoised_image.to("cpu").detach().numpy()
    clean_image = clean_image.to("cpu").detach().numpy()
    
    ssim = structural_similarity(denoised_image, clean_image, data_range=1, multichannel=is_multichannel)
    psnr = peak_signal_noise_ratio(denoised_image, clean_image, data_range=1)
    return ssim, psnr


def log(loss, ssim, psnr, stage, wandb_logger=None):
    if wandb_logger:
        wandb_logger.log({f"{stage}_loss": loss})
        wandb_logger.log({f"{stage}_ssim": ssim})
        wandb_logger.log({f"{stage}_psnr": psnr})



def main():
    config = load_config()
    print(OmegaConf.to_yaml(config))

    torch.set_default_device(config.device)
    print(f"Using {config.device}")

    wandb_logger = get_wandb_logger(config)

    with torch.no_grad():
        torch.cuda.empty_cache()

    model = net_theta.get_example_autoencoder(device=config.device)
    # model = full_pdhg_net.DynamicImageStaticPrimalDualNN(
    #     T=config.train_params.T,
    #     cnn_block=net_theta.UNet(),
    #     up_bound=config.train_params.up_bound,
    #     phase="training",
    #     device=config.device,
    # )

    loss_function = config.train_params.criterion
    optimizer_name = config.train_params.optimizer.name
    learning_rate = config.train_params.learning_rate

    criterion = get_criterion(loss_function)
    optimizer = get_optimizer(optimizer_name, learning_rate, model.parameters())
    
    train_loader, val_loader, test_loader = chest_xray_data.get_data_loaders_all(config, num_images=config.data_loader.num_images)

    device = config.device
    epochs = config.train_params.epochs
    is_multichannel = config.data_loader.is_multichannel
    save_model_every = config.save_model.epochs
    save_model_dir = config.save_model.dir

    for epoch in range(epochs):

        train_loss, train_ssim, train_psnr = train_one_epoch(model, train_loader, criterion, optimizer, device, is_multichannel)
        log(train_loss, train_ssim, train_psnr, "train", wandb_logger)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, SSIM: {train_ssim}, PSNR: {train_psnr}")

        if (epoch + 1) % save_model_every == 0:
            time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
            torch.save(model.state_dict(), f"{save_model_dir}_{time_str}/model_epoch_{epoch+1}.pt")

        val_loss, val_ssim, val_psnr = evaluate(model, val_loader, criterion, device, is_multichannel)
        log(val_loss, val_ssim, val_psnr, "val", wandb)
        print(f"Validation Loss: {val_loss}, SSIM: {val_ssim}, PSNR: {val_psnr}")

        with torch.no_grad():
            torch.cuda.empty_cache()

    print("Finished Training")

    with torch.no_grad():
        torch.cuda.empty_cache()

    test_loss, test_ssim, test_psnr = test(model, test_loader, criterion, config.device)
    log(test_loss, test_ssim, test_psnr, "test", wandb_logger)
    print(f"Test Loss: {test_loss}, SSIM: {test_ssim}, PSNR: {test_psnr}")

    wandb.finish()

    with torch.no_grad():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

# !python3 autoencoder.py
# !python3 autoencoder.py --config=config.yaml
# !python3 autoencoder.py --config=config.yaml --epochs=5
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data --wandb=False
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data --wandb=False --seed=42
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data --wandb=False --seed=42 --transform=transform
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data --wandb=False --seed=42 --transform=transform --model=model
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data --wandb=False --seed=42 --transform=transform --model=model --criterion=criterion
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data --wandb=False --seed=42 --transform=transform --model=model --criterion=criterion --optimizer=optimizer
# !python3 autoencoder.py --config=config.yaml --epochs=5 --batch_size=32 --learning_rate=0.001 --device=cuda --base_dir=data --wandb=False --seed=42 --transform=transform --model=model --criterion=criterion --optimizer=optimizer --train=train