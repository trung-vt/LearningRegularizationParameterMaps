import time
# NOTE: Importing torch the first time will always take a long time!
print(f"Importing torch in {__file__} ...")
import_torch_start_time = time.time() 
import torch
print(f"Importing torch took {time.time() - import_torch_start_time} seconds")