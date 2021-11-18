from math import sqrt

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dynamic_dataset import MovieLensDataset, my_collate_fn, my_collate_fn2, MyCollator, NamedMovieLensDataset
from globals import test_set_file, val_batch_size
from models import NCF
from models.AdvancedNCF import AdvancedNCF
from models.AttentionNCF import AttentionNCF
from models.NCF import load_model_state_and_params, load_model
from plots import plot_residuals, plot_stacked_residuals, plot_att_stats, plot_rated_items_counts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False
keep_att_stats = False


def evaluate_model(model: NCF):
    model.to(device)

    # load dataset
    test_dataset = NamedMovieLensDataset(test_set_file)
    print('Test size:', len(test_dataset))

    att_stats = None
    I = test_dataset.get_I()
    if visualize and isinstance(model, AttentionNCF):
        B = 4
        test_loader = DataLoader(test_dataset, batch_size=B, collate_fn=MyCollator(only_rated=True, with_names=True), shuffle=True)
    elif keep_att_stats and isinstance(model, AttentionNCF):
        att_stats = {'sum': pd.DataFrame(index=MovieLensDataset.get_sorted_item_names(), columns=MovieLensDataset.get_sorted_item_names(), data=np.zeros((I, I))),
                     'count': pd.DataFrame(index=MovieLensDataset.get_sorted_item_names(), columns=MovieLensDataset.get_sorted_item_names(), data=np.zeros((I, I), dtype=np.int32))}
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, collate_fn=MyCollator(only_rated=False, with_names=True))
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
            if visualize and isinstance(model, AttentionNCF):
                # get the input matrices and the target
                candidate_items, rated_items, user_matrix, y_batch, candidate_names, rated_names = data
                # forward model
                out = model(candidate_items.float().to(device), rated_items.float().to(device), user_matrix.float().to(device),
                            candidate_names=candidate_names, rated_names=rated_names, visualize=True)
            elif keep_att_stats and isinstance(model, AttentionNCF):
                # get the input matrices and the target
                candidate_items, rated_items, user_matrix, y_batch, candidate_names, rated_names = data
                # forward model
                out = model(candidate_items.float().to(device), rated_items.float().to(device), user_matrix.float().to(device),
                            att_stats=att_stats, candidate_names=candidate_names, rated_names=rated_names)
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

    if keep_att_stats and isinstance(model, AttentionNCF):
        plot_rated_items_counts(att_stats['count'], item_names=MovieLensDataset.get_sorted_item_names())
        plot_att_stats(att_stats, item_names=MovieLensDataset.get_sorted_item_names())
    else:
        fitted_values = np.concatenate(fitted_values, dtype=np.float64).reshape(-1)
        ground_truth = np.concatenate(ground_truth, dtype=np.float64).reshape(-1)

        # plot_fitted_vs_targets(fitted_values, ground_truth)
        plot_stacked_residuals(fitted_values, ground_truth)
        plot_residuals(fitted_values, ground_truth)


if __name__ == '__main__':
    model_file = '../models/AdvancedNCF_audio_meta.pt'

    # get metadata dim
    item_dim = MovieLensDataset.get_metadata_dim()

    # load model with correct layer sizes
    model = load_model(model_file, AdvancedNCF)
    print(model)

    # evaluate model on test set
    evaluate_model(model)
