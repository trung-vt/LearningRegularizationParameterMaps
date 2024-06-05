import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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


def check_pixel_value_range(images_folder:str) -> None:
    """
    Check the pixel value range of the images in the dataset.

    Parameters
    ----------
    images_folder : str
        the folder containing the images

    Returns
    -------
    None
    """
    images = os.listdir(images_folder)
    image = images[0]
    image = Image.open(f"{images_folder}/{image}")
    print(f"Image {image} has pixel values in the range [{np.min(image)}, {np.max(image)}]")


def check_image_dimensions(base_dir) -> None:
    """
    Check the min and max dimensions of the images in the dataset.

    Returns
    -------
    None
    """
    image_sizes = []
    datasets = ["train", "val", "test"]
    labels = ["NORMAL", "PNEUMONIA"]
    for dataset in datasets:
        for label in labels:
            for image_name in os.listdir(f"{base_dir}/{dataset}/{label}"):
                if not image_name.endswith(".jpeg"): continue
                image = Image.open(f"{base_dir}/{dataset}/{label}/{image_name}")
                image_sizes.append(image.size)
    image_sizes = np.array(image_sizes)
    max_width = np.max(image_sizes[:, 0])
    max_height = np.max(image_sizes[:, 1])
    min_width = np.min(image_sizes[:, 0])
    min_height = np.min(image_sizes[:, 1])
    print(f"Max width: {max_width}, max height: {max_height}")
    print(f"Min width: {min_width}, min height: {min_height}")

    expected_min_width = 100
    expected_min_height = 100
    assert min_width > expected_min_width, f"Expect all images to have width > {expected_min_width}. Got min width: {min_width}"
    assert min_height > expected_min_height, f"Expect all images to have height > {expected_min_height}. Got min height: {min_height}"