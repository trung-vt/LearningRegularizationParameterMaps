def get_config():
    return {
        "project_name": "SIDD",
        "architecture": "UNET-PDHG",
        "data": {
            "dataset": "Static Image Denoising Dataset",
            "original_dataset": "https://abdokamel.github.io/sidd/",
            "data_path": "data/sidd/images/patches",
            "train_num_samples": 9000,
            "val_num_samples": 512,
            "test_num_samples": 512,
            "patch_size": 256,
            "random_seed": 97,
        },
        "noise": {
            "min_sigma": 0.1,
            "max_sigma": 0.5,
            "random_seed": 42,
        },
        "unet": {
            "in_channels": 1,
            "out_channels": 2,
            "init_filters": 32,
            "n_blocks": 2,
            "activation": "LeakyReLU",
            "downsampling_kernel": (2, 2, 1),
            "downsampling_mode": "max",
            "upsampling_kernel": (2, 2, 1),
            "upsampling_mode": "linear_interpolation",
        },
        "pdhg": {
            # "up_bound": 0.5,
            "up_bound": 0,
            "T_train": 128,  # Higher T, NET does not have to try as hard? Less overfitting?
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "loss_function": "MSELoss",
            "epochs": 10_000,
            "batch_size": 1,
        },
        "evaluation": {
            "metrics": ["SSIM", "PSNR"],
        },
        "others": {
            "device": "cuda:0",
            "wandb_mode": "online",
            "save_epoch_wandb": 10_000,
            "save_epoch_local": 10,
            "save_dir": "models",
        }
    }