import torch
from torch.utils.data import Dataset
import numpy as np

from data.transform import crop_and_resize, add_noise
from data.chest_xray_load_images import load_images_chest_xray
from data.base_dataset import BaseDataset

# Code adapted from https://www.github.com/koflera/LearningRegularizationParameterMaps

class ChestXrayDataset(BaseDataset):
	def __init__(
		self, 
		data_path: str, 
		ids: list,
		resize_square = 256,
		sigma = (0.1, 0.5),  
		device: str = "cuda"
	):
		xray_images = load_images_chest_xray(data_path, ids)
		super().__init__(xray_images, resize_square, sigma, device)



# Code adapted from ...

def get_datasets(config):
    min_sigma = config["min_sigma"]
    max_sigma = config["max_sigma"]
    resize_square = config["resize_square"]
    device = config["device"]

    train_num_samples = config["train_num_samples"]
    train_ids = list(range(0, train_num_samples))
    dataset_train = ChestXrayDataset(
        data_path=config["train_data_path"],
        ids=train_ids,
        sigma=(min_sigma, max_sigma),
        resize_square=resize_square,
        # strides=[120, 120, 1],
        # patches_size=[120, 120, 1],
        # strides=[256, 256, 1], # stride < patch will allow overlapping patches, maybe good to blend the patches?
        # strides=[512, 512, 1],
        # patches_size=[512, 512, 1],
        device=device
    )

    val_num_samples = config["val_num_samples"]
    val_ids = list(range(0, val_num_samples))
    dataset_valid = ChestXrayDataset(
        data_path=config["val_data_path"],
        ids=val_ids,
        sigma=(min_sigma, max_sigma),
        resize_square=resize_square,
        # strides=[120, 120, 1],
        # patches_size=[120, 120, 1],
        # strides=[256, 256, 1], # stride < patch will allow overlapping patches, maybe good to blend the patches?
        # strides=[512, 512, 1],
        # patches_size=[512, 512, 1],
        device=device
    )

    test_num_samples = config["test_num_samples"]
    test_ids = list(range(0, test_num_samples))
    dataset_test = ChestXrayDataset(
        data_path=config["test_data_path"],
        ids=test_ids,
        sigma=(min_sigma, max_sigma),
        resize_square=resize_square,
        # strides=[120, 120, 1],
        # patches_size=[120, 120, 1],
        # strides=[256, 256, 1], # stride < patch will allow overlapping patches, maybe good to blend the patches?
        # strides=[512, 512, 1],
        # patches_size=[512, 512, 1],
        device=device
    )

    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of validation samples: {len(dataset_valid)}")
    print(f"Number of test samples: {len(dataset_test)}")

    return dataset_train, dataset_valid, dataset_test