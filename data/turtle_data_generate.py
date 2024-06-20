import torch
import os
from PIL import Image
from skimage.util import random_noise
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
# import multiprocessing # to count the number of CPU cores
import logging

# Configure logging to output to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


from data.transform import crop_and_resize, add_noise

class TurtleDataGenerator:
    def __init__(self, turtle_data_path='turtle_id_2022', size=256, dry_run=False, num_threads=4, sigmas=[0.1, 0.2, 0.3]):
        self.size = size
        self.dry_run = dry_run
        self.input_path = f'{turtle_data_path}/images'
        self.output_path_color = f'{turtle_data_path}/images_crop_resize_{size}'
        self.output_path_greyscale = f'{turtle_data_path}/images_crop_resize_{size}_greyscale'
        self.sigmas = sigmas
        self.output_paths_noisy = [f"{turtle_data_path}/images_crop_resize_{size}_greyscale_noisy_{str(sigma).replace('.', '_')}" for sigma in sigmas]
        
        # # Determine the number of CPU cores
        # num_cores = multiprocessing.cpu_count()
        # print(f"Number of CPU cores: {num_cores}")
        
        # # Decide number of threads based on the nature of the tasks
        # # For CPU-bound tasks
        # num_threads = num_cores
        # print(f"CPU-bound tasks, use few threads. num_threads = num_cores = {num_threads}")
        
        # # For I/O-bound tasks, you might want more threads
        # # num_threads = min(len(data), num_cores * 2)
        # # print(f"I/O-bound tasks, use many threads. num_threads = min(len(data), num_cores * 2) = {num_threads}")

        self.num_threads = num_threads
        print(f"Using {num_threads} threads")
        print("Using skimage.util.random_noise to add noise to images")


    # Function to process the list in chunks
    def process_list_in_chunks(self, data, num_chunks):
        chunk_size = len(data) // num_chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks

    def generate_cropped_and_resized_images(self):
        self.count = 0
        subfolders = os.listdir(self.input_path)
        
        # # Divide the list into chunks
        # chunks = self.process_list_in_chunks(subfolders, self.num_threads)
        
        # # Use ThreadPoolExecutor to process chunks in parallel
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
        #     print(f"Processing {len(subfolders)} subfolders in {len(chunks)} chunks")
        #     futures = [executor.submit(self.process_subfolders_chunk, chunk) for chunk in chunks]
            
        #     # Wait for all threads to complete and collect results
        #     for future in concurrent.futures.as_completed(futures):
        #         future.result()  # This will raise exceptions if any occurred
        
        print(f"Multiprocessing {len(subfolders)} subfolders in {self.num_threads} threads")
        pool = ThreadPool(self.num_threads)
        results = pool.map(self.process_subfolder, subfolders)
        
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
        
        # # images_noisy = [add_noise_PIL(img_greyscale, sigma) for sigma in self.sigmas]
        # images_noisy = [torch.tensor(img_greyscale, dtype=torch.float) / 255 for sigma in self.sigmas]
        # images_noisy = [add_noise(x, sigma) for x, sigma in zip(images_noisy, self.sigmas)]
        # images_noisy = [Image.fromarray((x * 255).numpy().astype(np.uint8)) for x in images_noisy]
        # images_noisy = [img, img_color, img_greyscale]
        # images_noisy = [np.array(img_greyscale, dtype=np.float32) / 255 for sigma in self.sigmas]
        # images_noisy = [img + np.random.normal(0, sigma, img.shape) for img, sigma in zip(images_noisy, self.sigmas)]
        # images_noisy = [np.clip(x, 0, 1) for x in images_noisy]
        # images_noisy = [Image.fromarray((x * 255).astype(np.uint8)) for x in images_noisy]
        images_noisy = [random_noise(np.array(img_greyscale), mode='gaussian', var=sigma**2, clip=True) for sigma in self.sigmas]
        images_noisy = [Image.fromarray((x * 255).astype(np.uint8)) for x in images_noisy]
        
        if not self.dry_run:
            # if image already exists, skip
            if not os.path.exists(output_image_path_color):
                img_color.save(output_image_path_color)
            if not os.path.exists(output_image_path_greyscale):
                img_greyscale.save(output_image_path_greyscale)
            for i, img_noisy in enumerate(images_noisy):
                output_image_path_noisy = f'{output_subfolder_greyscale_noisy[i]}/{image}'
                img_noisy.save(output_image_path_noisy)
                # if not os.path.exists(output_image_path_noisy):
                #     img_noisy.save(output_image_path_noisy)
        
        self.count += 1
