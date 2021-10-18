from math import sqrt

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MovieLensDataset
from globals import test_set_file, batch_size
from model import BasicNCF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_model(model: nn.Module):
    model.to(device)

    # load dataset
    test_dataset = MovieLensDataset(test_set_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.MSELoss(reduction='sum')   # don't average the loss as we shall do that ourselves for the whole epoch

    # Calculate val_loss and see if we need to stop
    model.eval()  # gradients "off"
    test_sum_loss = 0.0
    test_size = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            # get the item & user input and the target
            X_item_batch, X_user_batch, y_batch = data
            # forward model
            out = model(X_item_batch.float().to(device), X_user_batch.float().to(device))
            # calculate loss
            loss = criterion(out, y_batch.view(-1, 1).float().to(device))
            # accumulate validation loss
            test_sum_loss += loss.detach().item()
            test_size += len(y_batch)

    test_mse = test_sum_loss / test_size
    print(f'Test loss (MSE): {test_mse:.6f} - RMSE: {sqrt(test_mse):.6f}')


if __name__ == '__main__':
    model_file = '../models/3layers_model_066val.pt'

    # get metadata dim
    item_dim = MovieLensDataset.get_metadata_dim()

    # create model and load trained weights
    model = BasicNCF(item_dim, item_dim)
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    # evaluate model on test set
    evaluate_model(model)
