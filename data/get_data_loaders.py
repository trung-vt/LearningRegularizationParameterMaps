import torch

def get_data_loaders(config, get_datasets):

    dataset_train, dataset_valid, dataset_test = get_datasets(config)
    batch_size = config["batch_size"]
    device = config["device"]
    random_seed = config["random_seed"]

    # Create training dataloader
    # train_sampler = WeightedRandomSampler(dataset_train.samples_weights, len(dataset_train.samples_weights))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, 
        # sampler=train_sampler,
        generator=torch.Generator(device=device).manual_seed(random_seed),
        shuffle=True,
    )

    # Create validation dataloader 
    # val_sampler = WeightedRandomSampler(dataset_valid.samples_weights, len(dataset_valid.samples_weights))
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, 
        # sampler=val_sampler,
        generator=torch.Generator(device=device).manual_seed(random_seed),
        shuffle=False,
    )

    # Create test dataloader
    # test_sampler = WeightedRandomSampler(dataset_test.samples_weights, len(dataset_test.samples_weights))
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, 
        # sampler=test_sampler,
        generator=torch.Generator(device=device).manual_seed(random_seed),
        shuffle=False,
    )

    return (
        dataloader_train, 
        dataloader_valid, 
        dataloader_test,
    )