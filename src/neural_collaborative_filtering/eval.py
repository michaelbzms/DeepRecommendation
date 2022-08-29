import sys
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import ndcg_score, dcg_score

from globals import movie_text_info_file, full_matrix_file
from neural_collaborative_filtering.datasets.base import PointwiseDataset
from neural_collaborative_filtering.models.advanced_ncf import AttentionNCF
from neural_collaborative_filtering.models.base import NCF
from neural_collaborative_filtering.plots import plot_residuals, plot_stacked_residuals, plot_rated_items_counts, \
    plot_att_stats, visualize_attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False
keep_att_stats = True


def eval_ranking(samples_with_preds: pd.DataFrame, cutoff=10):
    """
    Parameter `samples_with_preds` should be the (userId, movieId, rating) DataFrame we are using elsewhere
    but with a new column `prediction` added with the model's prediction of the rating.

    This is not implementation agnostic but it was the only convenient way I found to implement the NDCG metric.
    TODO: make more generic, perhaps a wrapper class that should be extended.
    """
    ndcgs = []
    adj_ndcgs = []
    ignored = 0
    for userId, row in tqdm(samples_with_preds.groupby('userId').agg(list).iterrows(), total=len(samples_with_preds['userId'].unique()),
                            desc='Calculating NDCG', file=sys.stdout):
        if len(row['movieId']) <= 1:
            print("Warning: Found user with no interactions or only one ")
            ignored += 1
            continue

        # get y_pred from model in appropriate format
        y_pred = np.array(row['prediction'], dtype=np.float64)

        # get y_true from known test labels
        y_true = np.array(row['rating'], dtype=np.float64)

        # calculate ndcg for this user
        ndcg = ndcg_score([y_true], [y_pred], k=cutoff)

        # calculate custom NDCG for this user
        dcg = dcg_score([y_true], [y_pred], k=cutoff)
        ideal_dcg = dcg_score([y_true], [y_true], k=cutoff)
        worst_dcg = dcg_score([y_true], [5.0 - y_true], k=cutoff)

        if ideal_dcg == worst_dcg:
            ignored += 1
            continue     # nothing to do if all of them are ties...

        # min-max scaling formula
        adj_ndcg = (dcg - worst_dcg) / (ideal_dcg - worst_dcg)

        # append to ndcgs for all users
        ndcgs.append(ndcg)
        adj_ndcgs.append(adj_ndcg)

    # average ndcgs for all users to get a general estimate
    final_ndcg = np.mean(ndcgs)
    final_adj_ndcg = np.mean(adj_ndcgs)

    if ignored > 0:
        print('Ignored', ignored, 'samples because all their interactions were ties.')

    return final_ndcg, final_adj_ndcg


def eval_model(model: NCF, test_dataset: PointwiseDataset, batch_size, ranking, wandb=None, doplots=True):
    """
    Main logic for evaluating a model for our task. Other than the loss we also calculate TP, FP, FN and TN
    in order to calculate other metrics such as accuracy, recall and precision.
    """
    assert isinstance(test_dataset, PointwiseDataset), 'Should only be testing on pointwise datasets.'
    print('Test size:', len(test_dataset))

    # move model to GPU if available
    model.to(device)

    if visualize:
        batch_size = 1  # smaller batch size

    # define data loader (!) Important to use sequential sampler or the NDCG calculated will be wrong (!)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.use_collate())

    # get graphs if evaluating gnn (else it will be None)
    test_graph = test_dataset.get_graph(device)

    # movie info and att stats
    movie_info_df = pd.read_csv(movie_text_info_file, index_col=0)
    full_utility_matrix = pd.read_csv(full_matrix_file + '.csv')
    all_items = sorted(full_utility_matrix['movieId'].unique())
    if keep_att_stats and isinstance(model, AttentionNCF):
        n = len(all_items)
        att_stats = {
            'sum': np.zeros((n, n)),
            'count': np.zeros((n, n), dtype=np.int)
        }
        itemIDsPos = pd.Series(index=all_items, data=np.arange(n))
    else:
        att_stats = None
        itemIDsPos = None

    # Calculate val_loss and see if we need to stop
    model.eval()  # gradients "off"
    test_sum_loss = 0.0
    test_loss = 0.0
    fitted_values = []
    ground_truth = []
    extra_test_args = [] if test_graph is None else [test_graph]
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', file=sys.stdout):
            # forward model
            if isinstance(model, AttentionNCF):
                out, y_batch, candidate_items_IDs, rated_items_ids, att_weights = test_dataset.__class__.do_forward(model, batch, device, return_attention_weights=True)
            else:
                out, y_batch = test_dataset.__class__.do_forward(model, batch, device, *extra_test_args)
            # MSE only makes sense for regression task
            if not ranking:
                # calculate loss
                mse_loss = test_dataset.calculate_loss(out, y_batch.to(device))
                # accumulate test loss
                test_sum_loss += mse_loss.detach().item()
            # keep track of fitted values and their actual targets
            fitted_values.append(out.cpu().detach().numpy())
            ground_truth.append(y_batch.view(-1, 1).float().cpu().detach().numpy())
            if keep_att_stats and isinstance(model, AttentionNCF):
                candidate_pos = itemIDsPos.loc[candidate_items_IDs]
                rated_pos = itemIDsPos.loc[rated_items_ids]
                att_weights = att_weights.cpu().numpy()
                counts = np.array(att_weights > 0.0, dtype=np.int)
                for i, c in enumerate(candidate_pos):
                    att_stats['sum'][c, rated_pos] += att_weights[i, :]
                    att_stats['count'][c, rated_pos] += counts[i, :]
            if visualize and isinstance(model, AttentionNCF):
                visualize_attention(att_weights.cpu().numpy(),
                                    None,
                                    movie_info_df.loc[candidate_items_IDs]['primaryTitle'].values,
                                    movie_info_df.loc[rated_items_ids]['primaryTitle'].values)

    if keep_att_stats and isinstance(model, AttentionNCF):
        plot_att_stats(att_stats, movie_info_df.loc[itemIDsPos.index]['primaryTitle'].values)   # TODO: is correct?

    if not ranking:
        # MSE only makes sense for regression task
        test_loss = test_sum_loss / len(test_dataset)
        print(f'Test loss (MSE): {test_loss:.4f} - RMSE: {sqrt(test_loss):.4f}')

    # gather all predicted values and all ground truths
    fitted_values = np.concatenate(fitted_values, dtype=np.float64).reshape(-1)
    ground_truth = np.concatenate(ground_truth, dtype=np.float64).reshape(-1)

    # add fitted values to samples and calculate the NDCG
    test_dataset.samples['prediction'] = fitted_values
    ndcg5, adj_ndcg5 = eval_ranking(test_dataset.samples, cutoff=5)
    ndcg10, adj_ndcg10 = eval_ranking(test_dataset.samples, cutoff=10)
    ndcg20, adj_ndcg20 = eval_ranking(test_dataset.samples, cutoff=20)
    print(f'Test adj-NDCG@5 = {adj_ndcg5} - Test adj-NDCG@10 = {adj_ndcg10} - Test adj-NDCG@20 = {adj_ndcg20}')
    print(f'Test NDCG@5 = {ndcg5} - Test NDCG@10 = {ndcg10} - Test NDCG@20 = {ndcg20}')

    if wandb is not None:
        logs = {'test_ndcg@5': ndcg5, 'test_ndcg@10': ndcg10, 'test_ndcg@20': ndcg20,
                'test_adj_ndcg@5': adj_ndcg5, 'test_adj_ndcg@10': adj_ndcg10, 'test_adj_ndcg@20': adj_ndcg20}
        if not ranking:
            logs['test_loss'] = test_loss
        wandb.log(logs)

    if not ranking and doplots:
        # plot_fitted_vs_targets(fitted_values, ground_truth)
        plot_stacked_residuals(fitted_values, ground_truth)
        plot_residuals(fitted_values, ground_truth)
