from math import sqrt

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dynamic_dataset import MovieLensDataset, my_collate_fn, my_collate_fn2, MyCollator, NamedMovieLensDataset
from globals import test_set_file, val_batch_size
from models.AdvancedNCF import AdvancedNCF
from models.AttentionNCF import AttentionNCF
from plots import plot_residuals, plot_stacked_residuals

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = True


def evaluate_model(model: nn.Module):
    model.to(device)

    # load dataset
    test_dataset = NamedMovieLensDataset(test_set_file)
    print('Test size:', len(test_dataset))

    if visualize:
        B = 1
        test_loader = DataLoader(test_dataset, batch_size=B, collate_fn=MyCollator(with_names=True), shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, collate_fn=my_collate_fn)

    criterion = nn.MSELoss(reduction='sum')   # don't average the loss as we shall do that ourselves for the whole epoch

    # Calculate val_loss and see if we need to stop
    model.eval()  # gradients "off"
    test_sum_loss = 0.0
    test_size = 0
    fitted_values = []
    ground_truth = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            if visualize:
                # get the input matrices and the target
                candidate_items, rated_items, user_matrix, y_batch, candidate_names, rated_names = data
                # forward model
                out = model(candidate_items.float().to(device), rated_items.float().to(device), user_matrix.float().to(device),
                            candidate_names=candidate_names, rated_names=rated_names)
            else:
                # get the input matrices and the target
                candidate_items, rated_items, user_matrix, y_batch = data
                # forward model
                out = model(candidate_items.float().to(device), rated_items.float().to(device), user_matrix.float().to(device))
            # calculate loss
            loss = criterion(out, y_batch.view(-1, 1).float().to(device))
            # accumulate validation loss
            test_sum_loss += loss.detach().item()
            test_size += len(y_batch)
            # keep track of fitted values and their actual targets
            fitted_values.append(out.cpu().detach().numpy())
            ground_truth.append(y_batch.view(-1, 1).float().cpu().detach().numpy())

    test_mse = test_sum_loss / test_size
    print(f'Test loss (MSE): {test_mse:.6f} - RMSE: {sqrt(test_mse):.6f}')

    fitted_values = np.concatenate(fitted_values, dtype=np.float64).reshape(-1)
    ground_truth = np.concatenate(ground_truth, dtype=np.float64).reshape(-1)

    # plot_fitted_vs_targets(fitted_values, ground_truth)
    plot_stacked_residuals(fitted_values, ground_truth)
    plot_residuals(fitted_values, ground_truth)


if __name__ == '__main__':
    model_file = '../models/final_model.pt'

    # get metadata dim
    item_dim = MovieLensDataset.get_metadata_dim()

    # create model and load trained weights
    model = AttentionNCF(item_dim)
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    # evaluate model on test set
    evaluate_model(model)
