import os

from data.split_data import split_data

def split_turtle_data(
    turtle_data_path='data/turtle_id_2022/turtles-data/data',
    train_ratio=0.8, val_ratio=0.1, random_seed=42
):
    # Load the data
    subfolders = os.listdir(f"{turtle_data_path}/images")
    file_paths = []
    for subfolder in subfolders:
        file_paths += [f"{subfolder}/{file}" for file in os.listdir(f"{turtle_data_path}/images/{subfolder}")]
    # Split the data
    train, val, test = split_data(
        file_paths, 
        train_ratio=train_ratio, val_ratio=val_ratio, 
        random_seed=random_seed
    )
    
    # Save the split data
    with open(f"{turtle_data_path}/train.txt", "w") as f:
        f.write("\n".join(train))
    with open(f"{turtle_data_path}/val.txt", "w") as f:
        f.write("\n".join(val))
    with open(f"{turtle_data_path}/test.txt", "w") as f:
        f.write("\n".join(test))
