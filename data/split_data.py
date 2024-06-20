import random

def split_data(data, train_ratio=0.8, val_ratio=0.1, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    # shuffle data
    random.shuffle(data)
    
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    assert train_size + val_size < len(data)
    
    train_data = data[0 : train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size : ]
    
    return train_data, val_data, test_data