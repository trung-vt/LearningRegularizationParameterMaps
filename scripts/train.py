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


# Code adapted from https://www.github.com/koflera/LearningRegularizationParameterMaps


# Code taken from https://www.github.com/koflera/LearningRegularizationParameterMaps

def train_epoch(model, data_loader, optimizer, loss_func) -> float:
    running_loss = 0.
    running_psnr = 0.
    running_ssim = 0.
    num_batches = len(data_loader)
    # for sample in tqdm(data_loader): # tqdm helps show a nice progress bar
    for sample in data_loader:
        # loss, psnr, ssim = train_iteration(optimizer, model, loss_func, sample)

        optimizer.zero_grad(set_to_none=True)  # Zero your gradients for every batch! TODO: Why?
        noisy_image_5d, clean_image_5d = sample
        # print(f"noisy_image_5d.size(): {noisy_image_5d.size()}")
        # print(f"clean_image_5d.size(): {clean_image_5d.size()}")
        denoised_image_5d = model(noisy_image_5d)
        # print(f"denoised_image_5d.size(): {denoised_image_5d.size()}")
        loss = loss_func(denoised_image_5d, clean_image_5d)

        loss.backward()
        if loss.item() != loss.item():
            raise ValueError("NaN returned by loss function...")
        optimizer.step()

        denoised_image_2d = denoised_image_5d.squeeze(0).squeeze(0).squeeze(-1)
        clean_image_2d = clean_image_5d.squeeze(0).squeeze(0).squeeze(-1)
        psnr = PSNR(denoised_image_2d, clean_image_2d)
        ssim = SSIM(denoised_image_2d, clean_image_2d)

        running_loss += loss.item()
        running_psnr += psnr
        running_ssim += ssim
        
        wandb.log({"training_iter_loss": loss.item()})
        wandb.log({"training_iter PSNR": psnr})
        wandb.log({"training_iter SSIM": ssim})

        # Free up memory
        del loss 
        del denoised_image_5d # Delete output of model
        del denoised_image_2d # Delete auxiliary variable
        del clean_image_2d # Delete auxiliary variable
        del noisy_image_5d # Noisy image is generated each time so can delete it
        del clean_image_5d # TODO: Is this a copy that I can delete or a reference to the original?

    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches

    del running_loss
    del running_psnr
    del running_ssim
    del num_batches

    return avg_loss, avg_psnr, avg_ssim



def validate_epoch(model, data_loader, loss_func) -> float:
    running_loss = 0.
    running_psnr = 0.
    running_ssim = 0.
    num_batches = len(data_loader)
    # for sample in tqdm(data_loader): # tqdm helps show a nice progress bar
    for sample in data_loader:
        # loss, psnr, ssim = validate_iteration(model, loss_func, sample)

        noisy_image_5d, clean_image_5d = sample
        denoised_image_5d = model(noisy_image_5d)
        loss = loss_func(denoised_image_5d, clean_image_5d)

        denoised_image_2d = denoised_image_5d.squeeze(0).squeeze(0).squeeze(-1)
        clean_image_2d = clean_image_5d.squeeze(0).squeeze(0).squeeze(-1)
        psnr = PSNR(denoised_image_2d, clean_image_2d)
        ssim = SSIM(denoised_image_2d, clean_image_2d)

        running_loss += loss.item()
        running_psnr += psnr
        running_ssim += ssim
        
        wandb.log({"validation_iter_loss": loss.item()})
        wandb.log({"validation_iter PSNR": psnr})
        wandb.log({"validation_iter SSIM": ssim})

        # Free up memory
        del loss 
        del denoised_image_5d # Delete output of model
        del denoised_image_2d # Delete auxiliary variable
        del clean_image_2d # Delete auxiliary variable
        del noisy_image_5d # Noisy image is generated each time so can delete it
        del clean_image_5d # TODO: Is this a copy that I can delete or a reference to the original?

    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches

    del running_loss
    del running_psnr
    del running_ssim
    del num_batches

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

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=1, 
    #     generator=torch.Generator(device=DEVICE),
    #     shuffle=True)
    # data_loader_valid = torch.utils.data.DataLoader(
    #     dataset_valid, batch_size=1, 
    #     generator=torch.Generator(device=DEVICE),
    #     shuffle=False)


    init_wandb(config)

    # for epoch in range(start_epoch, num_epochs):
    for epoch in tqdm(range(start_epoch, num_epochs)):

        # Model training
        pdhg.train(True)
        training_loss, training_psnr, training_ssim = train_epoch(pdhg, data_loader_train, optimizer, loss_function)
        # training_loss, training_psnr, training_ssim = train_iteration(optimizer, pdhg, loss_function, sample=(noisy_image_data, clean_image_data))
        # print(f"Epoch {epoch+1} - TRAINING LOSS: {training_loss} - TRAINING PSNR: {training_psnr} - TRAINING SSIM: {training_ssim}")

        # Optional: Use wandb to log training progress
        wandb.log({"training_loss": training_loss})
        wandb.log({"training PSNR": training_psnr})
        wandb.log({"training SSIM": training_ssim})

        del training_loss
        del training_psnr
        del training_ssim

        pdhg.train(False)
        with torch.no_grad():
            torch.cuda.empty_cache()

            # Model validation
            validation_loss, validation_psnr, validation_ssim = validate_epoch(pdhg, data_loader_valid, loss_function)
            # validation_loss, validation_psnr, validation_ssim = validate_iteration(pdhg, loss_function, sample=(noisy_image_data, clean_image_data))
            # print(f"Epoch {epoch+1} - VALIDATION LOSS: {validation_loss} - VALIDATION PSNR: {validation_psnr} - VALIDATION SSIM: {validation_ssim}")
            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

            # Optional: Use wandb to log training progress
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation PSNR": validation_psnr})
            wandb.log({"validation SSIM": validation_ssim})

            torch.cuda.empty_cache()


        if (epoch+1) % save_epoch_local == 0:
            current_model_name = f"model_epoch_{epoch+1}"
            torch.save(pdhg, f"{model_states_dir}/{current_model_name}.pt")
            
            print(f"Epoch {epoch+1} - VALIDATION LOSS: {validation_loss} - VALIDATION PSNR: {validation_psnr} - VALIDATION SSIM: {validation_ssim}")

        # if (epoch+1) % save_epoch_wandb == 0:
        #     wandb.log_model(f"{model_states_dir}/{current_model_name}.pt", name=f"model_epoch_{epoch+1}")
            
        del validation_loss
        del validation_psnr
        del validation_ssim

        torch.cuda.empty_cache()


    # Save the entire model
    torch.save(pdhg, f"{model_states_dir}/final_model.pt")
    
    wandb.log_model(f"{model_states_dir}/final_model.pt", name=f"final_model")
    wandb.finish()
    
    with torch.no_grad():
        torch.cuda.empty_cache()

    return pdhg