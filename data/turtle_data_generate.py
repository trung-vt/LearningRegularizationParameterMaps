import os
from PIL import Image
from tqdm import tqdm
import concurrent.futures
# import multiprocessing # to count the number of CPU cores

from data.transform import crop_and_resize

class TurtleDataGenerator:
    def __init__(self, turtle_data_path='turtle_id_2022', size=256, dry_run=False, num_threads=4):
        self.size = size
        self.dry_run = dry_run
        self.input_path = f'{turtle_data_path}/images'
        self.output_path = f'{turtle_data_path}/images_crop_resize_{size}'
        
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


    # Function to process the list in chunks
    def process_list_in_chunks(self, data, num_chunks):
        chunk_size = len(data) // num_chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks

    def generate_cropped_and_resized_images(self):
        self.count = 0
        subfolders = os.listdir(self.input_path)
        
        # Divide the list into chunks
        chunks = self.process_list_in_chunks(subfolders, self.num_threads)
        
        # Use ThreadPoolExecutor to process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.process_subfolder, chunk) for chunk in chunks]
            
            # Wait for all threads to complete and collect results
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will raise exceptions if any occurred
        
        # for subfolder in tqdm(subfolders):
        #     self.process_subfolder(subfolder)
        print(f'Processed {self.count} images')
            
    def process_subfolder(self, subfolder):
        input_subfolder = f'{self.input_path}/{subfolder}'
        output_subfolder = f'{self.output_path}/{subfolder}'
        if not self.dry_run:
            os.makedirs(output_subfolder, exist_ok=True)
        images = os.listdir(input_subfolder)
        for image in images:
            self.process_image(image, input_subfolder, output_subfolder)
            
    def process_image(self, image, input_subfolder, output_subfolder):
        input_image_path = f'{input_subfolder}/{image}'
        output_image_path = f'{output_subfolder}/{image}'
        if not self.dry_run:
            img = Image.open(input_image_path)
            img = crop_and_resize(img, self.size)
            img.save(output_image_path)
        self.count += 1
            