from data import get_and_save_patches

def generate_patches():
    base_path="../data/sidd"
    dataset_type="medium"
    output_folder="patches"
    input_path=f"{base_path}/{dataset_type}/Data"
    output_path=f"{base_path}/{output_folder}"

    # Took about 5 minutes to generate from 320 images, 32 patches of size 256x256 from each image
    get_and_save_patches(input_path, output_path, type="GT", n_patches=32, patch_size=(256, 256), rand_seed=97)


# generate_patches()