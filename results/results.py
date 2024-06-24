import os
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import img2pdf

from data.transform import convert_to_tensor_4D, convert_to_PIL


class ResultGenerator:
    def __init__(self, model_path, min_lambda, max_lambda, num_lambdas, cmp_func, saving_denoised:bool, in_path:str, out_path:str, file_paths:dict, returning_denoised_PILs:bool=False):
        self.model = torch.load(model_path)
        self.model.eval()
        self.lambdas = np.linspace(min_lambda, max_lambda, num_lambdas)
        
        self.cmp_func = cmp_func
        self.saving_denoised = saving_denoised
        
        self.out_path = out_path
        
        self.returning_denoised_PILs = returning_denoised_PILs
        
        sigmas = list(file_paths.keys())[1:]
        original_file_paths = file_paths[0]
        self.sample_collection = []
        
        for sigma in sigmas:
            noisy_file_paths = file_paths[sigma]
            assert len(noisy_file_paths) == len(original_file_paths), f"len(noisy_file_paths) != len(original_file_paths)\nlen(noisy_file_paths): {len(noisy_file_paths)}, len(original_file_paths): {len(original_file_paths)}"
            for i in tqdm(range(len(noisy_file_paths))):
                noisy_file_path = noisy_file_paths[i]
                clean_file_path = original_file_paths[i % len(original_file_paths)]
                noisy_4d = convert_to_tensor_4D(np.array(Image.open(in_path + "/" + noisy_file_path)))
                clean_4d = convert_to_tensor_4D(np.array(Image.open(in_path + "/" + clean_file_path)))
                self.sample_collection.append((noisy_4d, clean_4d, noisy_file_path))
                
            
    def get_denoised_folder(self, noisy_path):
        extension = noisy_path.split(".")[-1]
        denoised_folder = self.out_path + "/" + noisy_path.replace(f".{extension}", "")
        if self.saving_denoised:
            os.makedirs(denoised_folder, exist_ok=True)
        return denoised_folder, extension
        
    def get_denoised_filename(self, denoised_folder, _lambda):
        _lambda = float(_lambda)
        # Change to string with exactly 3 decimal places and replace '.' with '_'
        _lambda = f"{_lambda:.3f}".replace('.', '_')
        denoised_filename = f"{denoised_folder}/lambda_{_lambda}"
        return denoised_filename


    def get_denoised_PIL(self, noisy_4d, clean_4d, denoised_folder, extension, _lambda):
        denoised_filename = self.get_denoised_filename(denoised_folder, _lambda)
        denoised_file = f"{denoised_filename}.{extension}"
        denoised_PIL = None
        
        if os.path.exists(denoised_file):
            denoised_PIL:Image = Image.open(denoised_file)
            denoised_4d:torch.tensor = convert_to_tensor_4D(denoised_PIL)
        else:
            denoised_5d:torch.tensor = self.model(noisy_4d.unsqueeze(0).to("cuda"), _lambda)
            assert len(denoised_5d.shape) == 5, f"Model output has unexpected shape {denoised_5d.shape}. Expected 5D tensor."
            denoised_4d:torch.tensor = denoised_5d.squeeze(0).cpu()
            del denoised_5d # Explicitly free up memory
        cmp_results = self.cmp_func(denoised_4d, clean_4d)
        if self.saving_denoised:
            if denoised_PIL is None:
                denoised_PIL:Image = convert_to_PIL(denoised_4d)
            denoised_PIL.save(denoised_file)
            denoised_pdf = f"{denoised_filename}.pdf"
            with open(denoised_pdf, "wb") as f:
                f.write(img2pdf.convert(denoised_file))
        del denoised_4d # Explicitly free up memory
        return denoised_PIL, cmp_results
            
        
    def brute_force_scalar_reg(self, sample):
        noisy_4d, clean_4d, noisy_path = sample
        assert noisy_4d.shape == clean_4d.shape, f"Noisy and clean images have different sizes!\nnoisy.shape: {noisy_4d.shape}, clean.shape: {clean_4d.shape}"

        df = pd.DataFrame(columns=["lambda", "MSE", "PSNR", "SSIM"])
        
        denoised_PILs = []
        denoised_folder, extension = self.get_denoised_folder(noisy_path)
        # extension = "PNG" # lossless format
        extension = "PDF" # to avoid rasterization when adding to latex?
        for _lambda in self.lambdas:
            denoised_PIL, cmp_results = self.get_denoised_PIL(noisy_4d, clean_4d, denoised_folder, extension, _lambda)
            if self.returning_denoised_PILs:
                denoised_PILs.append(denoised_PIL)
            df = pd.concat([df, pd.DataFrame([[_lambda, *cmp_results]], columns=df.columns)], ignore_index=True)
        
        df.set_index("lambda", inplace=True) # Set sigma as index
        df.sort_index(inplace=True)          # Sort by sigma
        df.to_csv(f"{denoised_folder}/results.csv", index=False)
        del noisy_4d, clean_4d # Explicitly free up memory
        return df, denoised_PILs
        
        
    def process_samples(self, num_threads:int=1):
        # Don't use too many threads, the computation is also done on the GPU which doesn't have a lot of memory?
        from multiprocessing.dummy import Pool as ThreadPool
        # https://stackoverflow.com/questions/2846653/how-do-i-use-threading-in-python
        pool = ThreadPool(num_threads)
        print(f"Multiprocessing {len(self.sample_collection)} samples in {num_threads} threads")
        results = pool.map(self.brute_force_scalar_reg, self.sample_collection)
        return results

