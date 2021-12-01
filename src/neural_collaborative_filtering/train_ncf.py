import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from neural_collaborative_filtering.models import NCF
from plots import plot_train_val_losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###########################################
#         Basic Training Loop             #
###########################################
def train_model(model: NCF, dataset_class, train_set_file, val_set_file,
                lr, weight_decay, batch_size, val_batch_size, early_stop,
                final_model_path, checkpoint_model_path='temp.pt', max_epochs=100,
                patience=5, stop_with_train_loss_instead=False,
                optimizer=None, save=True, writer: SummaryWriter=None):
    # torch.autograd.set_detect_anomaly(True)   # this slows down training
    model.to(device)

    # make sure we have compatible models and datasets (because batches and forward change)
    if not model.is_dataset_compatible(dataset_class):
        raise Exception('Model used is incompatible with this dataset.')

    # load dataset
    train_dataset = dataset_class(train_set_file)
    val_dataset = dataset_class(val_set_file)
    print('Training size:', len(train_dataset), ' - Validation size:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.use_collate())
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=val_dataset.use_collate())

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
        for batch in tqdm(train_loader, desc='Training'):
            # reset the gradients
            optimizer.zero_grad()
            # forward model
            out, y_batch = dataset_class.forward(model, batch, device)
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
            for batch in tqdm(val_loader, desc='Validating'):
                # forward model
                out, y_batch = dataset_class.forward(model, batch, device)
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