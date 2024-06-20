
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from data.transform import convert_to_tensor_4D
from data.turtle_data_generate import TurtleDataGenerator

def get_datasets(config):
    data_path = config["dataset"]
    with open(f"{data_path}/train.txt", "r") as f:
        train_files = f.read().splitlines()
    with open(f"{data_path}/val.txt", "r") as f:
        val_files = f.read().splitlines()
    with open(f"{data_path}/test.txt", "r") as f:
        test_files = f.read().splitlines()
        
    train_num_samples = config["train_num_samples"]
    val_num_samples = config["val_num_samples"]
    test_num_samples = config["test_num_samples"]
    
    train_files = train_files[:train_num_samples]
    val_files = val_files[:val_num_samples]
    test_files = test_files[:test_num_samples]
    
    size = config["resize_square"]
    num_threads = config["data_gen_num_threads"]
    sigmas = config["sigmas"]
    # Convert string "[00.1, 0.15, 0.2, 0.25, 0.3]" to list [0.1, 0.15, 0.2, 0.25, 0.3]
    sigmas = [float(sigma) for sigma in sigmas[1:-1].split(", ")]
    
    data_generator = TurtleDataGenerator(
        turtle_data_path='../data/turtle_id_2022/turtles-data/data', 
        size=size, num_threads=num_threads,
        sigmas=sigmas,
    )
    data_generator.generate_cropped_and_resized_images()
    
    train_dataset = TurtleDataset(data_path, train_files, size=size)
    val_dataset = TurtleDataset(data_path, val_files, size=size)
    test_dataset = TurtleDataset(data_path, test_files, size=size)
    
    return train_dataset, val_dataset, test_dataset
        

class TurtleDataset(torch.utils.data.Dataset):
    def get_image(self, img_path):
        img = Image.open(img_path)
        img_np = np.array(img)
        img_4d = convert_to_tensor_4D(img_np)
        return img_4d
    
    def __init__(self, data_path, files, sigmas=[0.1, 0.15, 0.2, 0.25, 0.3], size=512, device="cuda"):
        self.original = []
        for file in files:
            original_4d = self.get_image(f"{data_path}/images_crop_resize_{size}_greyscale/{file}")
            self.original.append(original_4d)
        self.original = torch.stack(self.original, dim=0).to(device)
            
        prefix = f"images_crop_resize_{size}_greyscale_noisy"
        self.images_pairs = []
        for sigma in sigmas:
            images_folder = f"{prefix}_{str(sigma).replace('.', '_')}"
            print(f"Loading images from {images_folder}")
            for i, file in tqdm(enumerate(files)):
                noisy_4d = self.get_image(f"{data_path}/{images_folder}/{file}").to(device)
                self.images_pairs.append((noisy_4d, i))
                
        self.device = device
        
    def __len__(self):
        return len(self.images_pairs)
    
    def __getitem__(self, idx):
        return (self.images_pairs[idx][0], self.original[self.images_pairs[idx][1]])
