def validate_iter(sample, model, loss_func, optimizer=None):
    noisy_5d, clean_5d = sample
    denoised_5d = model(noisy_5d)
    loss = loss_func(denoised_5d, clean_5d)

    denoised_2d = denoised_5d.squeeze(0).squeeze(0).squeeze(-1)
    clean_2d = clean_5d.squeeze(0).squeeze(0).squeeze(-1)
    psnr = PSNR(denoised_2d, clean_2d)
    ssim = SSIM(denoised_2d, clean_2d)

    wandb.log({"validation_iter_loss": loss.item()})
    wandb.log({"validation_iter PSNR": psnr})
    wandb.log({"validation_iter SSIM": ssim})

    # Free up memory
    del loss 
    del denoised_5d # Delete output of model
    del denoised_2d # Delete auxiliary variable
    del clean_2d # Delete auxiliary variable
    del noisy_5d # Noisy image is generated each time so can delete it
    del clean_5d # TODO: Is this a copy that I can delete or a reference to the original?
    
    return loss, psnr, ssim