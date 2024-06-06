def get_config():
    CHEST_XRAY_BASE_DATA_PATH = "../data/chest_xray"
    # CHEST_XRAY_BASE_DATA_PATH = "data/chest_xray"
    return {
        "project": "chest_xray",
        "dataset": CHEST_XRAY_BASE_DATA_PATH,
        "train_data_path": f"{CHEST_XRAY_BASE_DATA_PATH}/train/NORMAL",
        "val_data_path": f"{CHEST_XRAY_BASE_DATA_PATH}/val/NORMAL",
        "test_data_path": f"{CHEST_XRAY_BASE_DATA_PATH}/test/NORMAL",
        "train_num_samples": 2,
        "val_num_samples": 2,
        "test_num_samples": 1,

        # "patch": 512,
        # "stride": 512,
        "resize_square": 128, 
        "min_sigma": 0.5,
        "max_sigma": 0.5,
        "batch_size": 1,
        "random_seed": 42,

        "architecture": "UNET-PDHG",
        "unet_size": "64-128-256-3_pool-2_up-no_bottleneck",
        "activation": "LeakyReLU",

        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "loss_function": "MSELoss",

        # "up_bound": 0.5,
        "up_bound": 0,
        "T": 32,

        "epochs": 2,
        "device": "cuda",

        "save_epoch": 10,
        "save_dir": "tmp_2",
    }