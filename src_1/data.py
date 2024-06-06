# Code taken from https://www.github.com/koflera/LearningRegularizationParameterMaps

import os
import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in '{os.path.basename(__file__)}' ...")
import_torch_start_time = time.time() 
import torch
import_torch_end_time = time.time()
print(f"Importing torch in '{os.path.basename(__file__)}' took {import_torch_end_time - import_torch_start_time} seconds")

from torch.utils.data import Dataset, WeightedRandomSampler

from PIL import Image
import numpy as np
from tqdm import tqdm


# TODO: CHANGE THIS TO YOUR PATH
# NOTE: Windows uses \\ instead of /
def load_images_chest_xray(data_path: str, ids: list) -> list:
    files = os.listdir(data_path)
    jpeg_files = [f for f in files if f.endswith(".jpeg")]

    images = []
    for id in tqdm(ids):
        if id >= len(jpeg_files): continue
        # print(f"Loading image {id} from {data_path}")
        image = Image.open(os.path.join(data_path, jpeg_files[id]))
        images.append(image)
    
    return images


def crop_to_square(image: Image) -> Image:
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def crop_to_square_and_resize(image: Image, side_len: int) -> Image:
    image = crop_to_square(image)
    return image.resize(size=(side_len, side_len))


def add_noise(xf: torch.tensor, sigma) -> torch.tensor:
    std = torch.std(xf)
    mu = torch.mean(xf)

    x_centred = (xf  - mu) / std

    x_centred += sigma * torch.randn(xf.shape, dtype = xf.dtype)

    xnoise = std * x_centred + mu

    return xnoise


class DynamicImageStaticDenoisingDataset(Dataset):
	
	def __init__(
		self, 
		data_path: str, 
		ids: list,
		# scale_factor = 0.5, 
		# patches_size = None,
		# strides= None,
		resize_square = 120,
		sigma = (0.1, 0.5),  
		device: str = "cuda"
	):
		self.device = device
		# self.scale_factor = scale_factor
		self.resize_square = resize_square

		xray_images = load_images_chest_xray(data_path, ids)

		xf_list = []
		for image in xray_images:
			image = crop_to_square_and_resize(image, self.resize_square)
			image = image.convert('L') #convert to grey_scale
			image_data = np.asarray(image)
			xf = torch.tensor(image_data, dtype=torch.float)
			# Assume image is in [0, 255] range
			xf = xf / 255
			assert len(xf.size()) == 2, f"Expected 2D tensor, got {xf.size()}"
			xf = xf.unsqueeze(0) # Add channel dimension
			xf = xf.unsqueeze(-1) # Add time dimension. TODO: For legacy dynamic image code only. Will remove later.
			xf_list.append(xf)
		xf = torch.stack(xf_list, dim=0) # will have shape (mb, 1, Nx, Ny, Nt), where mb denotes the number of patches
		assert len(xf.size()) == 5, f"Expected 5D tensor, got {xf.size()}"
		assert xf.size(1) == 1, f"Expected 1 channel, got {xf.size(1)}"
		assert xf.size(2) == self.resize_square, f"Expected width (Nx) of {self.resize_square}, got {xf.size(-3)}"
		assert xf.size(3) == self.resize_square, f"Expected height (Ny) of {self.resize_square}, got {xf.size(-2)}"
		assert xf.size(4) == 1, f"Expected 1 time step, got {xf.size(-1)}"

		#create temporal TV vector to detect which patches contain the most motion
		xf_patches_tv = (xf[...,1:] - xf[...,:-1]).pow(2).sum(dim=[1,2,3,4]) #contains the TV for all patches
		
		#normalize to 1 to have a probability vector
		xf_patches_tv /= torch.sum(xf_patches_tv)
		
		#sort TV in descending order --> xfp_tv_ids[0] is the index of the patch with the most motion
		self.samples_weights = xf_patches_tv

		# TODO: Investigate
		# Change the values in samples_weights to be a range of integers from 0 to len(samples_weights)
		# Unless I do this, when I run on a set of identical images, it will give me an error:
		# RuntimeError: invalid multinomial distribution (encountering probability entry < 0)
		self.samples_weights = torch.arange(len(self.samples_weights))
		
		self.xf = xf
		self.len = xf.shape[0]
		
		self.sigma_min = sigma[0]
		self.sigma_max = sigma[1]
		
			
	def __getitem__(self, index):

		sigma = self.sigma_min + torch.rand(1) * ( self.sigma_max - self.sigma_min )

		x_noise = add_noise(self.xf[index], sigma)

		return (
			x_noise.to(device=self.device),
   			self.xf[index].to(device=self.device)
        )
		
	def __len__(self):
		return self.len
	


# Code adapted from ...

def get_datasets(config):
    min_sigma = config["min_sigma"]
    max_sigma = config["max_sigma"]
    resize_square = config["resize_square"]
    device = config["device"]

    train_num_samples = config["train_num_samples"]
    train_ids = list(range(0, train_num_samples))
    dataset_train = DynamicImageStaticDenoisingDataset(
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
    dataset_valid = DynamicImageStaticDenoisingDataset(
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
    dataset_test = DynamicImageStaticDenoisingDataset(
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



def get_dataloaders(config):

    dataset_train, dataset_valid, dataset_test = get_datasets(config)
    batch_size = config["batch_size"]
    device = config["device"]
    random_seed = config["random_seed"]

    # Create training dataloader
    # train_sampler = WeightedRandomSampler(dataset_train.samples_weights, len(dataset_train.samples_weights))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, 
        # sampler=train_sampler,
        generator=torch.Generator(
            #   device=device
        ).manual_seed(random_seed),
        shuffle=True,
    )

    # Create validation dataloader 
    # val_sampler = WeightedRandomSampler(dataset_valid.samples_weights, len(dataset_valid.samples_weights))
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, 
        # sampler=val_sampler,
        generator=torch.Generator(
            #   device=device
        ).manual_seed(random_seed),
        shuffle=False,
    )

    # Create test dataloader
    # test_sampler = WeightedRandomSampler(dataset_test.samples_weights, len(dataset_test.samples_weights))
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, 
        # sampler=test_sampler,
        generator=torch.Generator(
            #   device=device
        ).manual_seed(random_seed),
        shuffle=False,
    )

    return dataloader_train, dataloader_valid, dataloader_test