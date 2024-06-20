import matplotlib.pyplot as plt

from data.chest_xray_load_images import load_images_chest_xray

def test_load_images_chest_xray(stage="train", label="NORMAL"):
    CHEST_XRAY_BASE_DATA_PATH = "../data/chest_xray"
    for img in load_images_chest_xray(f"{CHEST_XRAY_BASE_DATA_PATH}/{stage}/{label}", [0]):
        print(img.size)
        plt.imshow(img, cmap='gray')
    plt.show();