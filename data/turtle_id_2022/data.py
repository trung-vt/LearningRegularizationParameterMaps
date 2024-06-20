from PIL import Image # to open images

def get_images_list(data_path="turtles-data", as_filepaths=True):
    """
    Get list of images in the dataset.

    Parameters
    ----------
    data_path : str
        Path to the dataset.

    as_filepaths : bool
        If True, returns list of filepaths. If False, returns list of image bytes.

    Returns
    -------
    list
        List of filepaths or image bytes.
    """

    