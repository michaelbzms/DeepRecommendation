from math import sqrt

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dynamic_movieLens_dataset import DynamicMovieLensDataset, MyCollator
from datasets.one_hot_dataset import OneHotMovieLensDataset
from globals import test_set_file, val_batch_size, USE_FEATURES
from neural_collaborative_filtering.models import NCF
from neural_collaborative_filtering.models.AttentionNCF import AttentionNCF
from neural_collaborative_filtering.models.BasicNCF import BasicNCF
from neural_collaborative_filtering.models.NCF import load_model
from plots import plot_residuals, plot_stacked_residuals, plot_att_stats, plot_rated_items_counts
from train import NCF_withfeatures_forward, NCF_onehot_forward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False
keep_att_stats = False


def evaluate_model(model: NCF, forward_function, dataset_class):
    model.to(device)

    # load dataset
    test_dataset = dataset_class(test_set_file)
    print('Test size:', len(test_dataset))

    att_stats = None
    I = dataset_class.get_number_of_items()
    if visualize and isinstance(model, AttentionNCF):
        B = 1
        test_loader = DataLoader(test_dataset, batch_size=B, collate_fn=MyCollator(only_rated=True, with_names=True), shuffle=True)
    elif keep_att_stats and isinstance(model, AttentionNCF):
        att_stats = {'sum': pd.DataFrame(index=dataset_class.get_sorted_item_names(), columns=dataset_class.get_sorted_item_names(), data=np.zeros((I, I))),
                     'count': pd.DataFrame(index=dataset_class.get_sorted_item_names(), columns=dataset_class.get_sorted_item_names(), data=np.zeros((I, I), dtype=np.int32))}
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, collate_fn=MyCollator(only_rated=False, with_names=True))
    else:
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, collate_fn=dataset_class.use_collate())

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
                # forward model
                out, y_batch = forward_function(model, data)
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
        plot_rated_items_counts(att_stats['count'], item_names=dataset_class.get_sorted_item_names())
        plot_att_stats(att_stats, item_names=dataset_class.get_sorted_item_names())
    else:
        fitted_values = np.concatenate(fitted_values, dtype=np.float64).reshape(-1)
        ground_truth = np.concatenate(ground_truth, dtype=np.float64).reshape(-1)

        # plot_fitted_vs_targets(fitted_values, ground_truth)
        plot_stacked_residuals(fitted_values, ground_truth)
        plot_residuals(fitted_values, ground_truth)


if __name__ == '__main__':
    model_file = '../models/final_model.pt'

    if USE_FEATURES:
        # get metadata dim
        item_dim = MovieLensDataset.get_item_feature_dim()

        # load model with correct layer sizes
        model = load_model(model_file, AttentionNCF)

        forward_function = NCF_withfeatures_forward
        dataset_class = MovieLensDataset
    else:
        # model = load_model(model_file, BasicNCF)

        state, _ = torch.load(model_file)
        model = BasicNCF(item_dim=OneHotMovieLensDataset.get_item_dim(),
                         user_dim=OneHotMovieLensDataset.get_user_dim())
        model.load_state_dict(state)

        forward_function = NCF_onehot_forward
        dataset_class = OneHotMovieLensDataset

        # make sure these are false
        visualize = False
        keep_att_stats = False

    print(model)

    # evaluate model on test set
    evaluate_model(model, forward_function, dataset_class)
