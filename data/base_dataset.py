import torch
from torch.utils.data import Dataset
import numpy as np

from data.transform import crop_and_resize, add_noise

# Code adapted from https://www.github.com/koflera/LearningRegularizationParameterMaps

class BaseDataset(Dataset):
	
	def __init__(
		self, 
        images: list,
		resize_square = 256,
		sigma = (0.1, 0.5), 
		device: str = "cuda"
	):
		self.device = device
		self.resize_square = resize_square

		xf_list = []
		for image in images:
			image = crop_and_resize(image, self.resize_square)
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

		# # TODO: Investigate
		# # Change the values in samples_weights to be a range of integers from 0 to len(samples_weights)
		# # Unless I do this, when I run on a set of identical images, it will give me an error:
		# # RuntimeError: invalid multinomial distribution (encountering probability entry < 0)
		# self.samples_weights = torch.arange(len(self.samples_weights))
		
		self.xf = xf
		self.len = xf.shape[0]
		
		self.sigma_min = sigma[0]
		self.sigma_max = sigma[1]
		
			
	def __getitem__(self, index):

		sigma = self.sigma_min + torch.rand(1) * ( self.sigma_max - self.sigma_min )

		x_noise = add_noise(self.xf[index], sigma)

		del sigma

		return (
			x_noise.to(device=self.device),
   			self.xf[index].to(device=self.device)
        )
		
	def __len__(self):
		return self.len