from data.sidd_load_images import load_images_SIDD
import matplotlib.pyplot as plt

def test_load_images_SIDD():
    SIDD_DATA_PATH = "../sidd/data/images/small/Data"
    images = load_images_SIDD(data_path=SIDD_DATA_PATH, ids=["0065"], from_npy_files=False)
    for img in images:
        print(img.size)
        plt.imshow(img)