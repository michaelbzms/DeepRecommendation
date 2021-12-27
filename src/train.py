import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from globals import train_set_file, val_set_file, weight_decay, lr, batch_size, max_epochs, early_stop, \
    stop_with_train_loss_instead, checkpoint_model_path, patience, dropout_rate, final_model_path, \
    val_batch_size, features_to_use, USE_FEATURES
from datasets.dynamic_movieLens_dataset import DynamicMovieLensDataset
from datasets.one_hot_dataset import OneHotMovieLensDataset
from neural_collaborative_filtering.models.AttentionNCF import AttentionNCF
from neural_collaborative_filtering.models.BasicMultimodalAttNCF import BasicMultimodalAttNCF
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
        dataset_class = DynamicMovieLensDataset

        # get F
        item_dim = dataset_class.get_item_feature_dim()

        # create model
        model = AttentionNCF(item_dim, dropout_rate=dropout_rate,
                             item_emb=256, user_emb=256, att_dense=5, mlp_dense_layers=[512, 256, 128])
        # model = AdvancedNCF(item_dim, item_emb=256, user_emb=256, mlp_dense_layers=[512, 256, 128], dropout_rate=dropout_rate)
    else:
        dataset_class = OneHotMovieLensDataset
        model = BasicMultimodalAttNCF(item_dim=dataset_class.get_number_of_items(),
                                      user_dim=dataset_class.get_number_of_users())
    print(model)

    # log training for later?
    now = datetime.now()
    writer = SummaryWriter('../runs/' + type(model).__name__ + '/' + now.strftime("%d_%m_%Y_%H_%M") + '/')   # unique per model
    hyperparams = {k: (torch.tensor(v) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    hyperparams['features_used'] = features_to_use
    writer.add_hparams(hyperparams, {})

    # train and save result
    train_model(model, dataset_class=dataset_class, train_set_file=train_set_file, val_set_file=val_set_file,
                lr=lr, weight_decay=weight_decay, batch_size=batch_size, val_batch_size=val_batch_size,
                early_stop=early_stop, final_model_path=final_model_path, checkpoint_model_path=checkpoint_model_path,
                max_epochs=max_epochs, patience=patience, stop_with_train_loss_instead=stop_with_train_loss_instead,
                writer=writer)
