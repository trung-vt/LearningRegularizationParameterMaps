
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from data.transform import convert_to_tensor_4D
from data.turtle_data_generate import TurtleDataGenerator
from config.load import get_sigmas


def get_original_folder(size):
    return f"images_crop_resize_{size}_greyscale"

def get_noisy_folder_prefix(size):
    return f"images_crop_resize_{size}_greyscale_noisy"

def get_noisy_folder(prefix, sigma):
    return f"{prefix}_{str(sigma).replace('.', '_')}"

def get_file_paths(base_path:str, type:str, num_samples:int, sigmas:list, size=int):
    """
    
    Parameters
    ----------
    num_samples : int
        Number of samples to load. If 0, load all samples.
    """
    with open(f"{base_path}/{type}.txt", "r") as f:
        files = f.read().splitlines()
    if num_samples != 0:
        files = files[:num_samples]
        
    file_paths = {}
        
    def load(sigma, folder):
        file_paths[sigma] = []
        for i in tqdm(range(len(files))):
            file = files[i]
            file_path = f"{folder}/{file}"
            file_paths[sigma].append(file_path)   
        
    original_folder = get_original_folder(size)
    print(f"Loading original image paths in {original_folder} ", end="")
    load(0, original_folder)
        
    prefix = get_noisy_folder_prefix(size)
    for sigma in sigmas:
        noisy_folder = get_noisy_folder(prefix, sigma)
        print(f"Loading noisy image paths sigma={sigma} in {noisy_folder} ", end="")
        load(sigma, noisy_folder)

    return file_paths



def get_datasets(config, generating_data:bool=False, device="cuda"):
    base_path = config["dataset"]
    size = config["resize_square"]
    sigmas = get_sigmas(config["sigmas"])
    
    if generating_data:
        num_threads = config["data_gen_num_threads"]
        data_generator = TurtleDataGenerator(
            turtle_data_path=base_path, 
            size=size, num_threads=num_threads,
            sigmas=sigmas,
        )
        data_generator.generate_cropped_and_resized_images()
    
    def get_dataset(type):
        num_samples = config[f"{type}_num_samples"]
        file_paths = get_file_paths(base_path, type, num_samples, sigmas, size)
        return TurtleDataset(base_path, file_paths, device=device)
        
    return get_dataset("train"), get_dataset("val"), get_dataset("test")
        

class TurtleDataset(torch.utils.data.Dataset):
    def get_img_4d(self, img_path):
        img = Image.open(img_path)
        img_np = np.array(img)
        img_4d = convert_to_tensor_4D(img_np)
        return img_4d
    
    def load(self, base_path, file_paths, img_list):
        for i in tqdm(range(len(file_paths))):
            file_path = base_path + "/" + file_paths[i]
            img_4d = self.get_img_4d(file_path).to(self.device)
            img_list.append(img_4d)
    
    def __init__(self, base_path, file_paths, device="cuda"):
        self.device = device
        sigmas = list(file_paths.keys())
        assert len(sigmas) > 1, "At least original and one noisy image group is required."
        assert sigmas[0] == 0, "First sigma must be 0 for original images."
        
        self.originals = []
        original_file_paths = file_paths[0]
        print(f"Loading original images ", end="")
        self.load(base_path, original_file_paths, self.originals)
        self.originals = torch.stack(self.originals, dim=0)

        self.noisies = []
        for j in range(1, len(sigmas)):
            noisy_file_paths = file_paths[sigmas[j]]
            print(f"Loading noisy images sigma={sigmas[j]} ", end="")
            self.load(base_path, noisy_file_paths, self.noisies)
        self.noisies = torch.stack(self.noisies, dim=0)


    def __len__(self): return len(self.noisies)

    def __getitem__(self, idx):
        noisy = self.noisies[idx]
        clean = self.originals[idx % len(self.originals)]
        return (noisy, clean)
