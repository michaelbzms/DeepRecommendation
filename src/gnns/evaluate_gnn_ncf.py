from math import sqrt

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_collaborative_filtering.models import NCF
from plots import plot_residuals, plot_stacked_residuals

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False


def eval_model(model: NCF, dataset_class, test_set_file, batch_size):
    model.to(device)

    # load dataset
    test_dataset = dataset_class(test_set_file)
    print('Test size:', len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=dataset_class.use_collate())

    test_graph = test_dataset.get_graph().to(device)

    criterion = nn.MSELoss(reduction='sum')  # don't average the loss as we shall do that ourselves for the whole epoch

    # Calculate val_loss and see if we need to stop
    model.eval()  # gradients "off"
    test_sum_loss = 0.0
    test_size = 0
    fitted_values = []
    ground_truth = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # forward model
            out, y_batch = dataset_class.forward(model, test_graph, batch, device)
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