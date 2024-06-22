import os
from PIL import Image
from skimage.util import random_noise
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import logging

# Configure logging to output to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


from data.transform import crop_and_resize, add_noise

print(f"Using new version 2 of turtle_data_generate.py")

class TurtleDataGenerator:
    def __init__(self, turtle_data_path='turtle_id_2022', size=256, dry_run=False, num_threads=4, sigmas=[0.1, 0.2, 0.3]):
        self.size = size
        self.dry_run = dry_run
        self.input_path = f'{turtle_data_path}/images'
        self.output_path_color = f'{turtle_data_path}/images_crop_resize_{size}'
        self.output_path_greyscale = f'{turtle_data_path}/images_crop_resize_{size}_greyscale'
        self.sigmas = sigmas
        self.output_paths_noisy = [f"{turtle_data_path}/images_crop_resize_{size}_greyscale_noisy_{str(sigma).replace('.', '_')}" for sigma in sigmas]
        self.num_threads = num_threads


    # Function to process the list in chunks
    def process_list_in_chunks(self, data, num_chunks):
        chunk_size = len(data) // num_chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks

    def generate_cropped_and_resized_images(self):
        self.count = 0
        subfolders = os.listdir(self.input_path)
        
        print(f"Multiprocessing {len(subfolders)} subfolders in {self.num_threads} threads")
        # https://stackoverflow.com/questions/2846653/how-do-i-use-threading-in-python
        pool = ThreadPool(self.num_threads)
        results = pool.map(self.process_subfolder, subfolders)
        
        # import tqdm
        # for subfolder in tqdm(subfolders):
        #     self.process_subfolder(subfolder)
        
        print(f'Processed {self.count} images')
        
    def process_subfolders_chunk(self, subfolders):
        try:
            for subfolder in subfolders:
                self.process_subfolder(subfolder)
            logging.info(f"Processing subfolders {subfolders}")
        except Exception as e:
            logging.error(f"Error processing subfolders {subfolders}: {e}")
            
    def process_subfolder(self, subfolder):
        input_subfolder = f'{self.input_path}/{subfolder}'
        output_subfolder_color = f'{self.output_path_color}/{subfolder}'
        output_subfolder_greyscale = f'{self.output_path_greyscale}/{subfolder}'
        output_subfolders_noisy = [f"{self.output_paths_noisy[i]}/{subfolder}" for i in range(len(self.output_paths_noisy))]
        if not self.dry_run:
            os.makedirs(output_subfolder_color, exist_ok=True)
            os.makedirs(output_subfolder_greyscale, exist_ok=True)
            for output_subfolder_noisy in output_subfolders_noisy:
                os.makedirs(output_subfolder_noisy, exist_ok=True)
        images = os.listdir(input_subfolder)
        for image in images:
            self.process_image(
                image, input_subfolder, 
                output_subfolder_color, 
                output_subfolder_greyscale,
                output_subfolders_noisy,
            )
            
    def process_image(
        self, image, input_subfolder, 
        output_subfolder_color,
        output_subfolder_greyscale,
        output_subfolder_greyscale_noisy,
    ):
        input_image_path = f'{input_subfolder}/{image}'
        output_image_path_color = f'{output_subfolder_color}/{image}'
        output_image_path_greyscale = f'{output_subfolder_greyscale}/{image}'
        img = Image.open(input_image_path)
        img_color = crop_and_resize(img, self.size)
        img_greyscale = img_color.convert('L')
        
        if not self.dry_run:
            # if image already exists, skip
            if not os.path.exists(output_image_path_color):
                img_color.save(output_image_path_color)
            if not os.path.exists(output_image_path_greyscale):
                img_greyscale.save(output_image_path_greyscale)
            for i, sigma in enumerate(self.sigmas):
                noisy_path = f'{output_subfolder_greyscale_noisy[i]}/{image}'
                if not os.path.exists(noisy_path):
                    img_noisy = random_noise(np.array(img_greyscale), mode='gaussian', var=sigma**2, clip=True)
                    img_noisy = Image.fromarray((img_noisy * 255).astype(np.uint8))
                    img_noisy.save(noisy_path)
        
        self.count += 1
