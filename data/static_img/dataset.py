import os

import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from .data_utils import extract_patches_2d


class StaticImageDenoisingDataset(Dataset):
	
	def __init__(
		self, 
		data_path: str, 
		ids: list,
		scale_factor = 0.5, 
		sigma=0.23,  
		patches_size = None,
		strides= None,
		force_recreate_data=True,
		device: str = "cuda"
	):
		self.device = device
		self.scale_factor = scale_factor

		ids = [str(x).zfill(2) for x in ids]

		xf_list = []

		k = 0
		for folder in os.listdir(data_path):
			# The first 4 characters of folder name is the image id
			img_id = folder[:4]
			if img_id not in ids:
				continue
			k += 1
			print(f'loading image id {img_id}, {k}/{len(ids)}')

			sample_path = os.path.join(data_path, folder)
			scale_factor_str = str(self.scale_factor).replace('.','_')
			npy_file = f"xf_scale_factor{scale_factor_str}.npy"

			# if file exists
			if os.path.exists(npy_file) and not force_recreate_data:
				xf = np.load(npy_file)
				xf = torch.tensor(xf, dtype=torch.float)
				xf = xf.unsqueeze(0) #TODO: Do we need to unsqueeze here when we will unsqueeze again later???
			else:
				xf = self.create_static_img(sample_path)
				np.save(os.path.join(sample_path, npy_file), xf)

			xf = xf.unsqueeze(0) / 255

			print(f"xf shape: {xf.shape}")

			if patches_size is not None:
				print(f"extracting patches of shape {patches_size}; strides {strides}")
				xfp = extract_patches_2d(xf.contiguous(), patches_size, stride=strides)
				print(f"xfp shape: {xfp.shape}")
				xf_list.append(xfp)
			
		if patches_size is not None:
			# will have shape (mb, 1, Nx, Ny, Nt), where mb denotes the number of patches
			# TODO: For static images, there is no Nt anymore???
			xf = torch.concat(xf_list,dim=0)
			
		else:
			xf = xf.unsqueeze(0) # TODO: Why do we need to unsqueeze again???
		
		#create temporal TV vector to detect which patches contain the most motion
		xfp_tv = (xf[...,1:] - xf[...,:-1]).pow(2).sum(dim=[1,2,3,4]) #contains the TV for all patches
		
		#normalize to 1 to have a probability vector
		xfp_tv /= torch.sum(xfp_tv)
		
		#sort TV in descending order --> xfp_tv_ids[0] is the index of the patch with the most motion
		self.samples_weights = xfp_tv
		
		self.xf = xf
		self.len = xf.shape[0]
		
		if isinstance(sigma, float):
			self.noise_level = 'constant'
			self.sigma = sigma

		elif isinstance(sigma, (tuple, list)):
			self.noise_level = 'variable'
			self.sigma_min = sigma[0]
			self.sigma_max = sigma[1]
		
		else:
			raise ValueError("Invalid sigma value provided, must be float, tuple or list.")

	def create_static_img(self, files_path: str):
		
		xf = []

		file_name = "GT_SRGB_010.PNG"
		image = Image.open(os.path.join(files_path, file_name))
		
		#resize
		Nx_,Ny_ = np.int(np.floor(self.scale_factor * image.width )), np.int(np.floor(self.scale_factor * image.height ))
		image = image.resize( (Nx_, Ny_) )
		
		#convert to grey_scale
		image = image.convert('L')
		image_data = np.asarray(image)
		xf.append(image_data)
			
		xf = np.stack(xf, axis=-1)
		print(f"xf shape: {xf.shape}")
		
		return torch.tensor(xf, dtype = torch.float)
			
	def __getitem__(self, index):

		std = torch.std(self.xf[index])
		mu = torch.mean(self.xf[index])

		x_centred = (self.xf[index]  - mu) / std

		if self.noise_level == 'constant':
			sigma = self.sigma
			
		elif self.noise_level == 'variable':
			sigma = self.sigma_min + torch.rand(1) * ( self.sigma_max - self.sigma_min )

		x_centred += sigma * torch.randn(self.xf[index].shape, dtype = self.xf[index].dtype)

		xnoise = std * x_centred + mu
  
		return (
			xnoise.to(device=self.device),
   			self.xf[index].to(device=self.device)
        )
		
	def __len__(self):
		return self.len