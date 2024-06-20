import os
from PIL import Image
from tqdm import tqdm

def load_images_chest_xray(data_path: str, ids: list) -> list:
    files = os.listdir(data_path)
    jpeg_files = [f for f in files if f.endswith(".jpeg")]

    images = []
    for id in tqdm(ids):
        if id >= len(jpeg_files): continue
        # print(f"Loading image {id} from {data_path}")
        image = Image.open(os.path.join(data_path, jpeg_files[id]))
        images.append(image)
    
    return images