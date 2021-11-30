import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from datasets.dynamic_dataset import MovieLensDataset

from datasets.dynamic_movieLens_dataset import DynamicMovieLensDataset
from datasets.one_hot_dataset import OneHotMovieLensDataset
from globals import train_set_file, val_set_file, weight_decay, lr, batch_size, max_epochs, early_stop, \
    stop_with_train_loss_instead, checkpoint_model_path, patience, dropout_rate, final_model_path, \
    val_batch_size, features_to_use, USE_FEATURES
from neural_collaborative_filtering.models.AttentionNCF import AttentionNCF
from neural_collaborative_filtering.models.BasicNCF import BasicNCF
from neural_collaborative_filtering.train_ncf import train_model


if __name__ == '__main__':
    # define optimizer and loss
    # For separate lrs:
    # optimizer = optim.Adam([
    #     {'params': model.item_embeddings.parameters(), 'lr': embeddings_lr},
    #     {'params': model.MLP.parameters(), 'lr': lr}
    # ], weight_decay=weight_decay)

    if USE_FEATURES:
        # get feature dim
        item_dim = MovieLensDataset.get_item_feature_dim()

        # create model
        model = AttentionNCF(item_dim, dropout_rate=dropout_rate,
                             item_emb=256, user_emb=256, att_dense=5, mlp_dense_layers=[512, 256, 128])
        # model = AdvancedNCF(item_dim, item_emb=256, user_emb=256, mlp_dense_layers=[512, 256, 128], dropout_rate=dropout_rate)

        dataset_class = DynamicMovieLensDataset
    else:
        model = BasicNCF(item_dim=OneHotMovieLensDataset.get_number_of_items(),
                         user_dim=OneHotMovieLensDataset.get_number_of_users())
        dataset_class = OneHotMovieLensDataset

    print(model)

    # log training for later?
    now = datetime.now()
    writer = SummaryWriter('../runs/' + type(model).__name__ + '/' + now.strftime("%d_%m_%Y_%H_%M") + '/')   # unique per model
    hyperparams = {k: (torch.tensor(v) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    hyperparams['features_used'] = features_to_use
    writer.add_hparams(hyperparams, {})

    # train and save result
    train_model(model, dataset_class=dataset_class, writer=writer)
