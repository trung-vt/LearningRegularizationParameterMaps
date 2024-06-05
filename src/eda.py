import os
from PIL import Image
import matplotlib.pyplot as plt

def visualise_images(images_folder:str, num_rows=3, num_cols=3, title=None) -> None:
    """
    Visualise some of the first images.

    Parameters
    ----------
    images_folder : str
        the folder containing the images
    num_rows : int
        the number of rows of subplots
    num_cols : int
        the number of columns of subplots
    title : str
        the title of the plot

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the folder does not exist
    """
    images = os.listdir(images_folder)

    plt.figure(figsize=(num_rows * 2, num_cols * 2))
    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        image = Image.open(f"{images_folder}/{images[i]}")
        plt.imshow(image, cmap="gray")
        plt.title(images[i], fontsize=8)
        plt.axis("off")

    if title is None: title = f"Images from {images_folder}"
    plt.suptitle(title, fontsize=10)
    plt.show();