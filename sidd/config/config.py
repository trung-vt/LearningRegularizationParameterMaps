config = {
    "project_name": "SIDD",
    "architecture": "UNET-PDHG",
    "data": {
        "dataset": "Static Image Denoising Dataset",
        "original_dataset": "",
        "data_path": "../data/sidd",
        "train_num_samples": 500,
        "val_num_samples": 50,
        "test_num_samples": 50,
        "resize_square": 256,
        "min_sigma": 0.1,
        "max_sigma": 0.5,
        "batch_size": 1,
        "random_seed": 42,
    },
    "unet": {
        "in_channels": 1,
        "out_channels": 2,
        "init_filters": 32,
        "n_blocks": 3,
        "activation": "LeakyReLU",
        "downsampling_kernel": (2, 2, 1),
        "downsampling_mode": "max",
        "upsampling_kernel": (2, 2, 1),
        "upsampling_mode": "linear_interpolation",
    },
    "pdhg": {
        # "up_bound": 0.5,
        "up_bound": 0,
        "T": 128,  # Higher T, NET does not have to try as hard? Less overfitting?
    },
    "training": {
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "loss_function": "MSELoss",
        "epochs": 10_000,
    },
    "evaluation": {
        "metrics": ["SSIM", "PSNR"],
    },
    "others": {
        "device": "cuda:0",
        "wandb_mode": "online",
        "save_epoch_wandb": 10_000,
        "save_epoch_local": 10,
        "save_dir": "tmp_2",
    }
}