from torchvision import datasets, transforms
from setup import DATA_PATH
import os

def load_data(transform):
    train_ds = datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_PATH, 'test'), transform)

    return train_ds, test_ds