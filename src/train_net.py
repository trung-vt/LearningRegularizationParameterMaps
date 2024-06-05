import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import time

def evaluate(model, val_loader, criterion, device="cuda", is_multichannel=False, wandb=None):
    model.eval()
    running_loss = 0.0
    val_ssim = 0.0
    val_psnr = 0.0
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

    val_loss = running_loss / len(val_loader)
    val_ssim = val_ssim / len(val_loader)
    val_psnr = val_psnr / len(val_loader)
    log(val_loss, val_ssim, val_psnr, "val", wandb)
    print(f"Validation Loss: {val_loss}, SSIM: {val_ssim}, PSNR: {val_psnr}")

def test_evaluate():
    model = nn.Conv2d(3, 3, 3, padding=1)
    val_loader = torch.utils.data.DataLoader(torch.randn(10, 3, 256, 256), batch_size=2)
    criterion = nn.MSELoss()
    evaluate(model, val_loader, criterion)


def train(model, train_loader, val_loader, criterion, optimizer, device="cuda", epochs=10, is_multichannel=False, wandb=None, save_model_every=2, save_model_dir="models"):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_ssim = 0.0
        train_psnr = 0.0
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

        train_loss = running_loss / len(train_loader)
        train_ssim = train_ssim / len(train_loader.dataset)
        train_psnr = train_psnr / len(train_loader.dataset)
        log(train_loss, train_ssim, train_psnr, "train", wandb)
        
        print(f"Epoch {epoch+1}, Loss: {train_loss}, SSIM: {train_ssim}, PSNR: {train_psnr}")

        time_str = time.strftime("%Y_%m_%d-%H_%M_%S")

        if (epoch + 1) % save_model_every == 0:
            torch.save(model.state_dict(), f"{save_model_dir}_{time_str}/model_epoch_{epoch+1}.pt")

        evaluate(model, val_loader, criterion, device, is_multichannel, wandb)

    print("Finished Training")


def test(model, test_loader, criterion, device="cuda", is_multichannel=False, wandb=None):
    model.eval()
    running_loss = 0.0
    test_ssim = 0.0
    test_psnr = 0.0
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


    test_loss = running_loss / len(test_loader)
    test_ssim = test_ssim / len(test_loader.dataset)
    test_psnr = test_psnr / len(test_loader.dataset)

    log(test_loss, test_ssim, test_psnr, "test", wandb)
    print(f"Test Loss: {test_loss}, SSIM: {test_ssim}, PSNR: {test_psnr}")


def get_metrics(denoised_image, clean_image, is_multichannel=False):
    # 5D to 2D. For example, (1, 1, 256, 256, 1) -> (256, 256)
    denoised_image = denoised_image.squeeze(0).squeeze(0).squeeze(-1)
    clean_image = clean_image.squeeze(0).squeeze(0).squeeze(-1)

    assert denoised_image.shape == clean_image.shape, f"Shape mismatch: denoised_image_shape={denoised_image.shape} does not match clean_image_shape={clean_image.shape}"

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


def log(loss, ssim, psnr, stage, wandb=None):
    if wandb:
        wandb.log({f"{stage}_loss": loss})
        wandb.log({f"{stage}_ssim": ssim})
        wandb.log({f"{stage}_psnr": psnr})