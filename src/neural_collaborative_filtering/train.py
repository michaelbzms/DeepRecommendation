import sys

import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from neural_collaborative_filtering.models.base import NCF
from neural_collaborative_filtering.util import load_model_state_and_params


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###########################################
#         Basic Training Loop             #
###########################################
def train_model(model: NCF, train_dataset, val_dataset,
                lr, weight_decay, batch_size, val_batch_size, early_stop,
                final_model_path, checkpoint_model_path='temp.pt', max_epochs=100,
                patience=5, stop_with_train_loss_instead=False, use_weighted_mse_for_training=False,
                optimizer=None, save=True, writer: SummaryWriter=None):
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

    # define data loaders - important to shuffle train set (!)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.__class__.use_collate())
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=val_dataset.__class__.use_collate())

    # define optimizer if not given
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # define loss function TODO: move to Dataset's do_forward()?
    criterion = nn.MSELoss(reduction='sum')  # don't average the loss as we shall do that ourselves for the whole epoch

    if use_weighted_mse_for_training:
        def weighted_mse_loss(input, target, weight):
            return torch.sum(weight * (input - target) ** 2)

        class_counts: pd.Series = train_dataset.get_class_counts()
        class_weights: pd.Series or None = 1.0 - (class_counts / class_counts.sum())   # TODO: are these the correct class weights?
        # class_weights: pd.Series or None = 100.0 / class_counts   # INS * 100
        print(class_weights)
        weighted_criterion = weighted_mse_loss
    else:
        weighted_criterion = None
        class_weights = None

    # get graphs if training gnn (else it will be None)
    train_graph = train_dataset.get_graph(device)
    val_graph = val_dataset.get_graph(device)

    # early stopping variables
    early_stop_times = 0
    least_running_loss = None
    previous_running_loss = None
    checkpoint_epoch = -1

    # keep track of training and validation losses
    monitored_metrics = {
        'train_loss': [],
        'val_loss': [],
    }

    for epoch in range(max_epochs):  # epoch
        print(f'\nEpoch {epoch + 1}')

        ##################
        #    Training    #
        ##################
        train_sum_loss = 0.0
        train_size = 0
        model.train()  # gradients "on"
        extra_train_args = [] if train_graph is None else [train_graph]
        for batch in tqdm(train_loader, desc='Training', file=sys.stdout):
            # reset the gradients
            optimizer.zero_grad()
            # forward model
            out, y_batch = train_dataset.__class__.do_forward(model, batch, device, *extra_train_args)
            # calculate loss
            if use_weighted_mse_for_training:
                loss = weighted_criterion(out, y_batch.view(-1, 1).float().to(device),
                                          torch.tensor(pd.Series(y_batch).map(class_weights).values).to(device))
            else:
                loss = criterion(out, y_batch.view(-1, 1).float().to(device))
            # backpropagation (compute gradients)
            loss.backward()
            # update weights according to optimizer
            optimizer.step()
            # accumulate train loss
            train_sum_loss += loss.detach().item()
            train_size += len(y_batch)  # Note: a little redundant doing this every epoch but it should be negligible

        train_loss = train_sum_loss / train_size
        monitored_metrics['train_loss'].append(train_loss)
        print(f'\nEpoch {epoch + 1}: Training {"weighted" if use_weighted_mse_for_training else ""} loss: {train_loss:.4f}')

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)

        ##################
        #   Validation   #
        ##################
        model.eval()  # gradients "off"
        val_sum_loss = 0.0
        val_size = 0
        extra_val_args = [] if train_graph is None else [val_graph]
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', file=sys.stdout):
                # forward model
                out, y_batch = val_dataset.__class__.do_forward(model, batch, device, *extra_val_args)
                # calculate loss
                loss = criterion(out, y_batch.view(-1, 1).float().to(device))
                # accumulate validation loss
                val_sum_loss += loss.detach().item()
                val_size += len(y_batch)

        val_loss = val_sum_loss / val_size
        monitored_metrics['val_loss'].append(val_loss)
        print(f'Validation loss: {val_loss:.4f}')

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)

        ######################
        #   Early Stopping   #
        ######################
        if early_stop:
            if least_running_loss is None or (not stop_with_train_loss_instead and val_sum_loss < least_running_loss) \
                    or (stop_with_train_loss_instead and train_sum_loss < least_running_loss):
                # always store the model with the least running loss achieved (saves kwargs as well)
                model.save_model(checkpoint_model_path)
                # write down latest checkpoint and best loss so far
                checkpoint_epoch = epoch
                least_running_loss = val_sum_loss if not stop_with_train_loss_instead else train_sum_loss
                # reset counter for patience
                early_stop_times = 0
            else:
                # increase early stop times only if loss increased from previous time (not the least one overall)
                if previous_running_loss is not None and (not stop_with_train_loss_instead and val_sum_loss > previous_running_loss) \
                        or (stop_with_train_loss_instead and train_sum_loss > previous_running_loss):
                    early_stop_times += 1
                    if float(early_stop_times).is_integer(): early_stop_times = int(early_stop_times)  # aesthetics
                else:
                    early_stop_times = max(0, early_stop_times - 1)

                if early_stop_times > patience:
                    print(f'Early stopping at epoch {epoch + 1}.')
                    print(f'Loading best model from checkpoint from epoch {checkpoint_epoch + 1} with loss: {least_running_loss / val_size:.4f}')
                    state, _ = load_model_state_and_params(checkpoint_model_path)  # ignore kwargs -> we know them
                    model.load_state_dict(state)
                    model.eval()
                    break
                else:
                    if epoch == max_epochs - 1:
                        # special case where we reached max_epochs and our current loss is not the best
                        print(f'Loss worsened in last epoch(s), loading best model from checkpoint from epoch {checkpoint_epoch}')
                        state, _ = load_model_state_and_params(checkpoint_model_path)  # ignore kwargs -> we know them
                        model.load_state_dict(state)
                        model.eval()

            print(f'Patience remaining: {patience - early_stop_times}')

            # update previous running loss
            previous_running_loss = val_sum_loss if not stop_with_train_loss_instead else train_sum_loss

    # save model (its weights)
    if save:
        print('Saving model...')
        model.save_model(final_model_path)
        print('Done!')

    if writer is not None:
        writer.flush()
        writer.close()

    return monitored_metrics
