import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from math import ceil

from glob import glob
from setup import DATA_PATH, IMAGE_HEIGHT, IMAGE_WIDTH
import os
from torchvision import transforms
from model.run_img import run_img

def evaluate_model(model: torch.nn.Module, test_data, batch_size):

    model = model.eval()

    # Making new Iterable
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    NUM_TEST_IMGS = len(test_data.imgs)
    NUM_TEST_BATCHES = ceil(NUM_TEST_IMGS/batch_size)
    iterable_ds_2 = iter(test_loader)

    accuracy = 0

    with torch.no_grad():
        predicted_classes = np.empty(NUM_TEST_IMGS)
        labels_all = np.empty(NUM_TEST_IMGS)
        model.cpu()
        for i in tqdm(range(NUM_TEST_BATCHES)): # NUM_BATCHES
            batch, labels = next(iterable_ds_2)
            batch = batch.cpu()
            labels = labels.cpu()
            pred = torch.softmax(model(batch), dim=0) # this will run softmax on each score, generating probabilities
            pred_classes = torch.max(pred, dim=1).indices # torch.max with dim=1 will run through every prediction and return the highest probability for that prediction.
            # the .indices will return the indices of thoses highest probabilities, which correspond to the classes.

            right_answers = (pred_classes == labels).sum() # for accuracy calculation
            accuracy += right_answers

            predicted_classes[i*batch_size:i*batch_size+batch_size] = pred_classes[:]
            labels_all[i*batch_size:i*batch_size+batch_size] = labels[:]

    print(f'\nAccuracy: {accuracy}/{NUM_TEST_IMGS} | {(accuracy*100/NUM_TEST_IMGS):.4f}%')

    print(confusion_matrix(labels_all, predicted_classes, labels=[0, 1, 2]))

def evaluate_single(model:torch.nn.Module, classes_list:list, device):
    """
        Performs an evaluation of the model running it against every image in a folder, individually.
        Prints the accuracy obtained.
    """

    model = model.eval()
    model.to(device)

    class_num = 0
    accuracy = 0
    num_elements = 0
    for class_str in classes_list:
        imgs = glob(os.path.join(DATA_PATH, 'test', class_str, '*.jpg'), recursive=True) + glob(os.path.join(DATA_PATH, 'test', class_str, '*.png'), recursive=True) + glob(os.path.join(DATA_PATH, 'test', class_str, '*.jpeg'), recursive=True) 

        for img in tqdm(imgs, desc=f'Testing Against Images of Class {class_str}'):
            num_elements += 1
            transform = transforms.Compose([transforms.Resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            predicted_class, _ = run_img(model, img, device, transform)

            if predicted_class == class_num:
                accuracy += 1
        
        class_num += 1
    print(f'\nAccuracy: {accuracy}/{num_elements} | {(accuracy*100/num_elements):.4f}%')