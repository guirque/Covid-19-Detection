from tqdm import tqdm


def train(model, num_epochs, num_batches, device, optimizer, scheduler, dataloader, loss):
    # Loop
    for epoch in range(num_epochs):
        iterable_ds = iter(dataloader) # reinitialize batches whenever we run a new epoch
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
            try:
                batch, labels = next(iterable_ds) # we get the images (batch) and corresponding labels. That's returned by __getitem__, it seems.
                batch = batch.to(device=device)
                labels = labels.to(device=device)
                #print(model.cl1.weight.device, batch.device)
                final_outputs = model(batch)

                loss_results = loss(final_outputs, labels)

                optimizer.zero_grad()
                loss_results.backward()
                optimizer.step()

                scheduler.step()
            except StopIteration:
                print(f'Exception Encountered: batch {i}/{num_batches} in epoch {epoch}')
