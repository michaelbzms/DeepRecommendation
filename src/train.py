import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from content_providers.dynamic_profiles_provider import DynamicProfilesProvider
from content_providers.fixed_profiles_provider import FixedProfilesProvider
from content_providers.one_hot_provider import OneHotProvider
from globals import train_set_file, val_set_file, weight_decay, lr, batch_size, max_epochs, early_stop, \
    stop_with_train_loss_instead, checkpoint_model_path, patience, dropout_rate, final_model_path, \
    val_batch_size, features_to_use, USE_FEATURES, use_weighted_mse_for_training, ranking_train_set_file, \
    ranking_val_set_file
from neural_collaborative_filtering.datasets.base import PointwiseDataset
from neural_collaborative_filtering.datasets.dynamic_datasets import DynamicPointwiseDataset

from neural_collaborative_filtering.datasets.fixed_datasets import FixedPointwiseDataset, FixedRankingDataset
from neural_collaborative_filtering.models.advanced_ncf import AttentionNCF, AdvancedNCF
from neural_collaborative_filtering.models.basic_ncf import BasicNCF
from neural_collaborative_filtering.plots import plot_train_val_losses
from neural_collaborative_filtering.train import train_model


if __name__ == '__main__':
    # define optimizer and loss
    # For separate lrs:
    # optimizer = optim.Adam([
    #     {'params': model.item_embeddings.parameters(), 'lr': embeddings_lr},
    #     {'params': model.MLP.parameters(), 'lr': lr}
    # ], weight_decay=weight_decay)

    if USE_FEATURES:
        # get feature dim
        # dpp = DynamicProfilesProvider()
        # training_dataset = DynamicPointwiseDataset(train_set_file, dpp)
        # val_dataset = DynamicPointwiseDataset(val_set_file, dpp)
        #
        # # get F
        # item_dim = dpp.get_item_feature_dim()
        #
        # # create model
        # model = AttentionNCF(item_dim, dropout_rate=dropout_rate,
        #                      item_emb=128, user_emb=128, att_dense=None, mlp_dense_layers=[200])
        # # # model = AdvancedNCF(item_dim, item_emb=256, user_emb=256, mlp_dense_layers=[512, 256, 128], dropout_rate=dropout_rate)

        fixed_provider = FixedProfilesProvider()
        training_dataset = FixedPointwiseDataset(train_set_file, content_provider=fixed_provider)
        val_dataset = FixedPointwiseDataset(val_set_file, content_provider=fixed_provider)

        model = BasicNCF(item_dim=fixed_provider.get_item_feature_dim(),
                         user_dim=fixed_provider.get_item_feature_dim(),
                         item_emb=128, user_emb=128, mlp_dense_layers=[256, 128])

        # fixed_provider = FixedProfilesProvider()
        # training_dataset = FixedRankingDataset(ranking_train_set_file, content_provider=fixed_provider)
        # val_dataset = FixedRankingDataset(ranking_val_set_file, content_provider=fixed_provider)
        #
        # model = BasicNCF(item_dim=fixed_provider.get_item_feature_dim(),
        #                  user_dim=fixed_provider.get_item_feature_dim(),
        #                  item_emb=128, user_emb=128, mlp_dense_layers=[256, 128])
    else:
        onehot_provider = OneHotProvider()
        training_dataset = FixedRankingDataset(ranking_train_set_file, content_provider=onehot_provider)
        val_dataset = FixedRankingDataset(ranking_val_set_file, content_provider=onehot_provider)

        model = BasicNCF(item_dim=onehot_provider.get_num_items(),
                         user_dim=onehot_provider.get_num_users(),
                         item_emb=128, user_emb=128, mlp_dense_layers=[256, 128])
    print(model)

    # log training for later?
    now = datetime.now()
    writer = SummaryWriter('../runs/' + type(model).__name__ + '/' + now.strftime("%d_%m_%Y_%H_%M") + '/')   # unique per model
    hyperparams = {k: (torch.tensor(v) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    hyperparams['features_used'] = features_to_use
    writer.add_hparams(hyperparams, {})

    # train and save result
    monitored_metrics = train_model(model, train_dataset=training_dataset, val_dataset=val_dataset,
                                    lr=lr, weight_decay=weight_decay, batch_size=batch_size, val_batch_size=val_batch_size,
                                    early_stop=early_stop, final_model_path=final_model_path, checkpoint_model_path=checkpoint_model_path,
                                    max_epochs=max_epochs, patience=patience, stop_with_train_loss_instead=stop_with_train_loss_instead,
                                    use_weighted_mse_for_training=use_weighted_mse_for_training, writer=writer)

    # plot and save losses
    plot_train_val_losses(monitored_metrics['train_loss'], monitored_metrics['val_loss'])
