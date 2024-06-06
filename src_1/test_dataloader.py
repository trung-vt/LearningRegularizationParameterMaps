
import matplotlib.pyplot as plt

from data import get_dataloaders, get_datasets
from config import get_config

def test_dataset():
    dataset_train, dataset_valid, dataset_test = get_datasets(get_config())
    x, y = dataset_train[0]
    print(f"Sample {1}")
    print(f"x size: {x.size()}")
    print(f"y size: {y.size()}")
    plt.subplot(1, 2, 1)
    plt.imshow(x.squeeze(0).squeeze(0).squeeze(-1).to("cpu"), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(y.squeeze(0).squeeze(0).squeeze(-1).to("cpu"), cmap='gray')
    plt.axis('off')
    plt.show();

test_dataset()



def test_dataloader():
    dataloader_train, dataloader_valid, dataloader_test = get_dataloaders(get_config())
    for i, (x, y) in enumerate(dataloader_train):
        print(f"Batch {i+1}")
        print(f"x size: {x.size()}")
        print(f"y size: {y.size()}")
        plt.subplot(1, 2, 1)
        plt.imshow(x.squeeze(0).squeeze(0).squeeze(-1).to("cpu"), cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(y.squeeze(0).squeeze(0).squeeze(-1).to("cpu"), cmap='gray')
        plt.axis('off')
        plt.show();
        if i == 5:
            break

test_dataloader()