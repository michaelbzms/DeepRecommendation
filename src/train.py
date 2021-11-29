import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from datasets.dynamic_dataset import MovieLensDataset, my_collate_fn, my_collate_fn2
from datasets.one_hot_dataset import OneHotMovieLensDataset
from globals import train_set_file, val_set_file, weight_decay, lr, batch_size, max_epochs, early_stop, \
    stop_with_train_loss_instead, checkpoint_model_path, patience, dropout_rate, final_model_path, \
    val_batch_size, features_to_use, USE_FEATURES
from models import NCF
from models.AdvancedNCF import AdvancedNCF
from models.AttentionNCF import AttentionNCF
from models.BasicNCF import BasicNCF
from plots import plot_train_val_losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


""" 
Define different kind of forwards here 
"""


def NCF_withfeatures_forward(model: NCF, data):
    # get the input matrices and the target
    candidate_items, rated_items, user_matrix, y_batch = data
    # forward model
    out = model(candidate_items.float().to(device), rated_items.float().to(device), user_matrix.float().to(device))
    return out, y_batch


def NCF_onehot_forward(model: NCF, data):
    # get the input matrices and the target
    item_vec, user_vec, y_batch = data
    # forward model
    out = model(item_vec.float().to(device), user_vec.float().to(device))
    return out, y_batch


def train_model(model: NCF, forward_function, dataset_class,
                save=True, optimizer=None, writer: SummaryWriter=None):
    # torch.autograd.set_detect_anomaly(True)   # this slows down training
    model.to(device)

    # load dataset
    train_dataset = dataset_class(train_set_file)
    val_dataset = dataset_class(val_set_file)
    print('Training size:', len(train_dataset), ' - Validation size:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.use_collate())
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=val_dataset.use_collate())

    # define optimizer and loss
    # For separate lrs:
    # optimizer = optim.Adam([
    #     {'params': model.item_embeddings.parameters(), 'lr': embeddings_lr},
    #     {'params': model.MLP.parameters(), 'lr': lr}
    # ], weight_decay=weight_decay)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='sum')   # don't average the loss as we shall do that ourselves for the whole epoch

    early_stop_times = 0
    least_running_loss = None
    checkpoint_epoch = -1
    train_losses = []
    val_losses = []
    for epoch in range(max_epochs):                   # epoch
        """ training """
        train_sum_loss = 0.0
        train_size = 0
        model.train()  # gradients "on"
        for data in tqdm(train_loader, desc='Training'):                     # batch
            # reset the gradients
            optimizer.zero_grad()
            # forward model
            out, y_batch = forward_function(model, data)
            # calculate loss
            loss = criterion(out, y_batch.view(-1, 1).float().to(device))
            # backpropagation (compute gradients)
            loss.backward()
            # update weights according to optimizer
            optimizer.step()
            # accumulate train loss
            train_sum_loss += loss.detach().item()
            train_size += len(y_batch)     # Note: a little redundant doing this every epoch but it should be negligible

        train_loss = train_sum_loss / train_size
        train_losses.append(train_loss)
        print(f'\nEpoch {epoch + 1}: Training loss: {train_loss:.6f}')

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)

        """ validation """
        # Calculate val_loss and see if we need to stop
        model.eval()   # gradients "off"
        val_sum_loss = 0.0
        val_size = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc='Validating'):
                # forward model
                out, y_batch = forward_function(model, data)
                # calculate loss
                loss = criterion(out, y_batch.view(-1, 1).float().to(device))
                # accumulate validation loss
                val_sum_loss += loss.detach().item()
                val_size += len(y_batch)

        val_loss = val_sum_loss / val_size
        val_losses.append(val_loss)
        print(f'Validation loss: {val_loss:.6f}')

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)

        if early_stop:
            if least_running_loss is None or (not stop_with_train_loss_instead and val_sum_loss < least_running_loss) \
                    or (stop_with_train_loss_instead and train_sum_loss < least_running_loss):
                model.save_model(checkpoint_model_path)   # saves kwargs as well
                checkpoint_epoch = epoch
                least_running_loss = val_sum_loss if not stop_with_train_loss_instead else train_sum_loss
                early_stop_times = 0  # reset
            else:
                if early_stop_times >= patience:
                    print(f'Early stopping at epoch {epoch + 1}.')
                    print(f'Loading best model from checkpoint from epoch {checkpoint_epoch + 1} with loss: {least_running_loss / val_size:.6f}')
                    state, _ = NCF.load_model_state_and_params(checkpoint_model_path)  # ignore kwargs -> we know them
                    model.load_state_dict(state)
                    # Old way: model.load_state_dict(torch.load(checkpoint_model_path))
                    model.eval()
                    break
                else:
                    early_stop_times += 1
                    if epoch == max_epochs - 1:
                        print(f'Loss worsened in last epoch(s), loading best model from checkpoint from epoch {checkpoint_epoch}')
                        state, _ = NCF.load_model_state_and_params(checkpoint_model_path)  # ignore kwargs -> we know them
                        model.load_state_dict(state)
                        model.eval()

    # save model (its weights)
    if save:
        print('Saving model...')
        model.save_model(final_model_path)
        print('Done!')

    if writer is not None:
        writer.flush()
        writer.close()

    # plot and save losses
    plot_train_val_losses(train_losses, val_losses)

    return model


if __name__ == '__main__':
    if USE_FEATURES:
        # get feature dim
        item_dim = MovieLensDataset.get_item_feature_dim()

        # create model
        model = AttentionNCF(item_dim, dropout_rate=dropout_rate,
                             item_emb=256, user_emb=256, att_dense=5, mlp_dense_layers=[512, 256, 128])
        # model = AdvancedNCF(item_dim, item_emb=256, user_emb=256, mlp_dense_layers=[512, 256, 128], dropout_rate=dropout_rate)

        forward_function = NCF_withfeatures_forward
        dataset_class = MovieLensDataset
    else:
        model = BasicNCF(item_dim=OneHotMovieLensDataset.get_item_dim(),
                         user_dim=OneHotMovieLensDataset.get_user_dim())
        forward_function = NCF_onehot_forward
        dataset_class = OneHotMovieLensDataset

    print(model)

    # log training for later?
    now = datetime.now()
    writer = SummaryWriter('../runs/' + type(model).__name__ + '/' + now.strftime("%d_%m_%Y_%H_%M") + '/')   # unique per model
    hyperparams = {k: (torch.tensor(v) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    hyperparams['features_used'] = features_to_use
    writer.add_hparams(hyperparams, {})

    # train and save result
    train_model(model, writer=writer, forward_function=forward_function, dataset_class=dataset_class)
