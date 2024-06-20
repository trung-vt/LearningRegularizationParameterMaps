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
