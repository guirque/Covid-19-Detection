from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from setup import BATCH_SIZE

def load_dataset(dataset):
    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=3
    )

    return iter(loader)

def train(model, num_epochs, num_batches, device, optimizer, scheduler, dataset, loss, num_imgs):
    # Loop
    for epoch in range(num_epochs):
        accuracy = 0

        #iterable_ds = iter(dataloader) # reinitialize batches whenever we run a new epoch
        iterable_ds = load_dataset(dataset)

        for i in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
            try:
                batch, labels = next(iterable_ds) # we get the images (batch) and corresponding labels. That's returned by __getitem__, it seems.
                batch = batch.to(device=device)
                labels = labels.to(device=device)

                final_outputs = model(batch)

                loss_results = loss(final_outputs, labels)

                optimizer.zero_grad()
                loss_results.backward()
                optimizer.step()

                scheduler.step()

                # Evaluating
                pred = torch.softmax(final_outputs, dim=0) 
                pred_classes = torch.max(pred, dim=1).indices
                right_answers = (pred_classes == labels).sum()
                accuracy += right_answers

            except StopIteration:
                print(f'Exception Encountered: batch {i}/{num_batches} in epoch {epoch}')
        print(f"Epoch {epoch} | Accuracy: {(accuracy*100/num_imgs):.4f}%")