import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from math import ceil

def evaluate_model(model, test_data, batch_size):

    # Making new Iterable
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
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
