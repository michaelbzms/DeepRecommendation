import sys

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_collaborative_filtering.datasets.base import PointwiseDataset, RankingDataset
from neural_collaborative_filtering.eval import eval_ranking
from neural_collaborative_filtering.models.base import NCF
from neural_collaborative_filtering.util import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###########################################
#         Basic Training Loop             #
###########################################
def train_model(model: NCF, train_dataset, val_dataset: PointwiseDataset,
                lr, weight_decay, batch_size, val_batch_size, early_stop,
                final_model_path='final_model.pt', checkpoint_model_path='temp.pt', max_epochs=100,
                patience=3, max_patience=5, optimizer=None, ndcg_cutoff=10, wandb=None, num_workers=0):
    """
    Main logic for training a model. Hyperparameters (e.g. lr, batch_size, etc) as arguments.
    On each loop (epoch), we forward the model on the training_dataset and backpropagate the loss
    in mini-batch fashion. On each epoch, we also calculate the loss on the validation set which
    we use for early stopping with a standard patience scheme.

    Logic for how to call forward() on the given model is expected to be in the dataset's do_forward()
    instead of here so that this code does not have to change for different model-dataset combos.
    """
    # For Debug: this slows down training but detects errors
    # torch.autograd.set_detect_anomaly(True)

    # move model to GPU if available
    model.to(device)

    # make sure we have compatible models and datasets (because batches and forward change)
    if not model.is_dataset_compatible(train_dataset.__class__) or not model.is_dataset_compatible(val_dataset.__class__):
        raise Exception('Model used is incompatible with this dataset.')

    print('Training size:', len(train_dataset), ' - Validation size:', len(val_dataset))

    if num_workers > 0:
        print(f'Using {num_workers} worker processes for training data loader.')

    # define data loaders
    # Note: Important to shuffle train set (!) - do NOT shuffle val set (if we do the NDCG calculation will be wrong)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.use_collate(), num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=val_dataset.use_collate())

    # define optimizer if not given
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # get graphs if training gnn (else it will be None)
    train_graph = train_dataset.get_graph(device)
    val_graph = val_dataset.get_graph(device)

    # early stopping variables
    early_stop_times = 0
    best_running_val_loss_or_ndcg = None
    previous_running_loss_or_ndcg = None
    checkpoint_epoch = -1
    best_val_loss = None
    best_ndcg = -1
    starting_max_patience = max_patience

    # keep track of training and validation losses
    monitored_metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_ndcg': []
    }

    # logs
    if wandb is not None:
        wandb.watch(model)

    for epoch in range(max_epochs):  # epoch
        w_str = ''
        if isinstance(train_dataset, RankingDataset):
            train_dataset.w = schedule_w(epoch + 1)
            w_str = f' (w = {train_dataset.w})'
        print(f'\nEpoch {epoch + 1}{w_str}')

        ##################
        #    Training    #
        ##################
        train_sum_loss = 0.0
        model.train()  # gradients "on"
        extra_train_args = [] if train_graph is None else [train_graph]
        y_preds = []
        for batch in tqdm(train_loader, desc='Training', file=sys.stdout):
            # reset the gradients
            optimizer.zero_grad()
            # forward model
            out, y_true_or_out2 = train_dataset.__class__.do_forward(model, batch, device, *extra_train_args)
            # calculate loss
            loss = train_dataset.calculate_loss(out, y_true_or_out2.to(device))
            # backpropagation (compute gradients)
            loss.backward()
            # update weights according to optimizer
            optimizer.step()
            # accumulate train loss
            train_sum_loss += loss.detach().item()
            # accumulate predictions if pointwise
            if isinstance(train_dataset, PointwiseDataset):
                y_preds.append(out.cpu().detach().numpy())
        train_loss = train_sum_loss / len(train_dataset)
        monitored_metrics['train_loss'].append(train_loss)
        print(f'Training loss: {train_loss:.4f}')

        ##################
        #   Validation   #
        ##################
        model.eval()  # gradients "off"
        val_loss = val_sum_loss = 0.0
        extra_val_args = [] if train_graph is None else [val_graph]
        y_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', file=sys.stdout):
                # forward model
                out, y_true_or_out2 = val_dataset.__class__.do_forward(model, batch, device, *extra_val_args)
                if isinstance(train_dataset, PointwiseDataset):  # if doing point-wise learning
                    # calculate loss
                    loss = val_dataset.calculate_loss(out, y_true_or_out2.to(device))
                    # accumulate validation loss
                    val_sum_loss += loss.detach().item()
                # accumulate predictions
                y_preds.append(out.cpu().detach().numpy())
        if isinstance(train_dataset, PointwiseDataset):
            val_loss = val_sum_loss / len(val_dataset)
            monitored_metrics['val_loss'].append(val_loss)

        # gather all predictions, add them to samples and calculate the NDCG
        y_preds = np.concatenate(y_preds, dtype=np.float64).reshape(-1)
        val_dataset.samples['prediction'] = y_preds    # overwriting previous is ok
        val_ndcg, val_adj_ndcg = eval_ranking(val_dataset.samples, cutoff=ndcg_cutoff)
        monitored_metrics['val_ndcg'].append(val_ndcg)
        if isinstance(train_dataset, PointwiseDataset):
            print(f'Validation loss: {val_loss:.4f}', end=' - ')
        print(f'Validation NDCG@{ndcg_cutoff}: {val_ndcg:.4f}, adj-NDCG@{ndcg_cutoff}: {val_adj_ndcg:.4f}')

        # keep track of max NDCG
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg

        # log training and validation metrics
        if wandb is not None:
            logs = {'train_loss': train_loss,
                    f'val_ndcg@{ndcg_cutoff}': val_ndcg,
                    f'val_adj_ndcg@{ndcg_cutoff}': val_adj_ndcg,
                    'epoch': epoch + 1}
            if isinstance(train_dataset, PointwiseDataset):      # only for point-wise learning
                logs['val_loss'] = val_loss
            elif isinstance(train_dataset, RankingDataset):
                logs['neg_sampling_w'] = train_dataset.w
            wandb.log(logs)

        ######################
        #   Early Stopping   #
        ######################
        if early_stop:
            if best_running_val_loss_or_ndcg is None or \
                    (isinstance(train_dataset, PointwiseDataset) and val_loss < best_running_val_loss_or_ndcg) or \
                    (isinstance(train_dataset, RankingDataset) and val_ndcg > best_running_val_loss_or_ndcg):
                # always store the model with the least running loss achieved (saves kwargs as well)
                model.save_model(checkpoint_model_path)
                # write down latest checkpoint and best loss so far
                checkpoint_epoch = epoch
                if isinstance(train_dataset, RankingDataset):  # if doing pair-wise learning
                    best_running_val_loss_or_ndcg = val_ndcg              # early stop using val_ndcg
                else:
                    best_running_val_loss_or_ndcg = val_loss              # else use normal val loss
                    best_val_loss = val_loss
                # reset counter for patience
                early_stop_times = 0
                # reset max patience
                max_patience = starting_max_patience
            else:
                # increase early stop times only if loss increased from previous time (not the least one overall)
                if previous_running_loss_or_ndcg is not None and \
                        (isinstance(train_dataset, PointwiseDataset) and val_loss > previous_running_loss_or_ndcg) or \
                        (isinstance(train_dataset, RankingDataset) and val_ndcg < previous_running_loss_or_ndcg):
                    early_stop_times += 1
                    if float(early_stop_times).is_integer(): early_stop_times = int(early_stop_times)  # aesthetics
                else:
                    early_stop_times = max(0, early_stop_times - 1)

                # decrease max_patience anyway if loss was worse than the best loss achieved overall
                max_patience -= 1

                if early_stop_times > patience or max_patience <= 0:
                    print(f'Early stopping at epoch {epoch + 1}.')
                    print(f'Loading best model from checkpoint from epoch {checkpoint_epoch + 1} with val {"loss" if isinstance(train_dataset, PointwiseDataset) else "ndcg"}: {best_running_val_loss_or_ndcg:.4f}')
                    state, _ = load_model(checkpoint_model_path)  # ignore kwargs -> we know them
                    model.load_state_dict(state)
                    model.eval()
                    break
                elif epoch == max_epochs - 1:
                    # special case where we reached max_epochs and our current loss is not the best
                    print(f'Val loss/ndcg worsened in last epoch(s), loading best model from checkpoint from epoch {checkpoint_epoch + 1}.')
                    state, _ = load_model(checkpoint_model_path)  # ignore kwargs -> we know them
                    model.load_state_dict(state)
                    model.eval()

            print(f'Patience remaining: {patience - early_stop_times}')

            # update previous running loss / ndcg
            previous_running_loss_or_ndcg = val_loss if isinstance(train_dataset, PointwiseDataset) else val_ndcg

    if wandb is not None:
        best_logs = {'best_ndcg@10': best_ndcg}
        if best_val_loss is not None:
            best_logs['best_val_loss'] = best_val_loss
        wandb.log(best_logs)

    # save model (its weights)
    if final_model_path is not None:
        print('Saving model...')
        model.save_model(final_model_path)
        print('Done!')

    return monitored_metrics


def schedule_w(epoch, break_points=(4, 8, 12, 18, 24)):
    break_points = sorted(break_points)
    assert len(break_points) >= 5, 'Invalid args'
    if epoch < break_points[0]:
        return 0.0
    elif epoch < break_points[1]:
        return 0.5
    elif epoch < break_points[2]:
        return 1
    elif epoch < break_points[3]:
        return 1.5
    elif epoch < break_points[3]:
        return 2
    else:
        return 3
