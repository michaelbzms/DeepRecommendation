from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from globals import train_set_file, val_set_file, weight_decay, lr, batch_size, max_epochs, early_stop, \
    stop_with_train_loss_instead, checkpoint_model_path, patience, dropout_rate, final_model_path, \
    val_batch_size
from neural_collaborative_filtering.models.gnn_ncf import GAT_NCF, GAT_NCF_Multimodal
from graph_datasets.movielens_gnn_dataset import MovieLensGNNDataset
from neural_collaborative_filtering.train import train_model
from globals import mask_target_edges_when_training


if __name__ == '__main__':
    dataset_class = MovieLensGNNDataset
    model = GAT_NCF(dropout_rate=dropout_rate)
    print(model)

    # define optimizer and loss
    # For separate lrs:
    # optimizer = optim.Adam([
    #     {'params': model.gnn_convs.parameters(), 'lr': embeddings_lr},
    #     {'params': model.MLP.parameters(), 'lr': lr},
    #     {'params': model.item_embeddings.parameters(), 'lr': lr},
    #     {'params': model.user_embeddings.parameters(), 'lr': lr},
    # ], weight_decay=weight_decay)
    optimizer = None

    # log training for later?
    now = datetime.now()
    writer = SummaryWriter('../runs/' + type(model).__name__ + '/' + now.strftime("%d_%m_%Y_%H_%M") + '/')   # unique per model
    # hyperparams = {k: (torch.tensor(v) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    # writer.add_hparams(hyperparams, {})

    # train and save result
    train_model(model, train_dataset=dataset_class(train_set_file), val_dataset=dataset_class(val_set_file),
                lr=lr, optimizer=optimizer, weight_decay=weight_decay, batch_size=batch_size, val_batch_size=val_batch_size,
                early_stop=early_stop, final_model_path=final_model_path, checkpoint_model_path=checkpoint_model_path,
                max_epochs=max_epochs, patience=patience, stop_with_train_loss_instead=stop_with_train_loss_instead,
                writer=writer)
