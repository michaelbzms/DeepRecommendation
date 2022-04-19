from datetime import datetime
import wandb

from content_providers.dynamic_profiles_provider import DynamicProfilesProvider
from content_providers.fixed_profiles_provider import FixedProfilesProvider
from content_providers.graph_providers import OneHotGraphProvider, ProfilesGraphProvider
from content_providers.one_hot_provider import OneHotProvider
from neural_collaborative_filtering.datasets.dynamic_datasets import DynamicPointwiseDataset, DynamicRankingDataset
from neural_collaborative_filtering.datasets.fixed_datasets import FixedPointwiseDataset, FixedRankingDataset
from neural_collaborative_filtering.datasets.gnn_datasets import GraphPointwiseDataset, GraphRankingDataset
from neural_collaborative_filtering.models.advanced_ncf import AttentionNCF
from neural_collaborative_filtering.models.basic_ncf import BasicNCF
from neural_collaborative_filtering.models.gnn_ncf import NGCF
from neural_collaborative_filtering.plots import plot_train_val_losses
from neural_collaborative_filtering.train import train_model
from globals import train_set_file, val_set_file, weight_decay, lr, batch_size, max_epochs, early_stop, \
    stop_with_train_loss_instead, checkpoint_model_path, patience, dropout_rate, final_model_path, \
    val_batch_size, ranking_train_set_file, \
    ranking_val_set_file


def prepare_fixedinput_ncf(ranking=False, use_features=False):
    """
    Set up the model and datasets to train an NCF model on fixed input.
    """
    if use_features:
        # content provider
        cp = FixedProfilesProvider()
        # model
        model = BasicNCF(item_dim=cp.get_item_feature_dim(),
                         user_dim=cp.get_item_feature_dim(),
                         item_emb=128, user_emb=128,
                         mlp_dense_layers=[256, 128],
                         dropout_rate=dropout_rate)
    else:
        # content provider
        cp = OneHotProvider()
        # model
        model = BasicNCF(item_dim=cp.get_num_items(),
                         user_dim=cp.get_num_users(),
                         item_emb=128, user_emb=128,
                         mlp_dense_layers=[256, 128],
                         dropout_rate=dropout_rate)
    # datasets
    if ranking:
        training_dataset = FixedRankingDataset(ranking_train_set_file, content_provider=cp)
        val_dataset = FixedRankingDataset(ranking_val_set_file, content_provider=cp)
    else:
        training_dataset = FixedPointwiseDataset(train_set_file, content_provider=cp)
        val_dataset = FixedPointwiseDataset(val_set_file, content_provider=cp)

    return model, training_dataset, val_dataset


def prepare_attention_ncf(ranking=False):
    """
    Set up the model and datasets to train the attention NCF model on dynamic input
    (i.e. dynamic user profiles).
    """
    # content provider
    dpp = DynamicProfilesProvider()
    # model
    model = AttentionNCF(dpp.get_item_feature_dim(), dropout_rate=dropout_rate,
                         item_emb=128, user_emb=128,
                         att_dense=None,
                         mlp_dense_layers=[256])

    # datasets
    if ranking:
        training_dataset = DynamicRankingDataset(ranking_train_set_file, dynamic_provider=dpp)
        val_dataset = DynamicRankingDataset(ranking_val_set_file, dynamic_provider=dpp)
    else:
        training_dataset = DynamicPointwiseDataset(train_set_file, dynamic_provider=dpp)
        val_dataset = DynamicPointwiseDataset(val_set_file, dynamic_provider=dpp)

    return model, training_dataset, val_dataset


def prepare_graph_ncf(ranking=False, use_features=False):
    """
    Set up the model and datasets to train a graph NCF model on fixed input.
    """

    if use_features:
        # content provider
        gcp = ProfilesGraphProvider(train_set_file)
        # model
        model = NGCF(item_dim=gcp.get_item_dim(),
                     user_dim=gcp.get_user_dim(),
                     gnn_hidden_layers=[64, 64],
                     node_emb=64,
                     mlp_dense_layers=[256, 128],
                     dropout_rate=dropout_rate)
    else:
        # content provider
        gcp = OneHotGraphProvider(train_set_file)
        # model
        model = NGCF(item_dim=gcp.get_item_dim(),
                     user_dim=gcp.get_user_dim(),
                     gnn_hidden_layers=[64, 64],
                     node_emb=64,
                     mlp_dense_layers=[256, 128],
                     dropout_rate=dropout_rate)
    # datasets
    if ranking:
        training_dataset = GraphRankingDataset(ranking_train_set_file, graph_content_provider=gcp)
        val_dataset = GraphRankingDataset(ranking_val_set_file, graph_content_provider=gcp)
    else:
        training_dataset = GraphPointwiseDataset(train_set_file, graph_content_provider=gcp)
        val_dataset = GraphPointwiseDataset(val_set_file, graph_content_provider=gcp)

    return model, training_dataset, val_dataset


if __name__ == '__main__':
    use_features = False

    # prepare model, train and val datasets
    model, training_dataset, val_dataset = prepare_fixedinput_ncf(ranking=False, use_features=use_features)
    # model, training_dataset, val_dataset = prepare_attention_ncf(ranking=False)
    # model, training_dataset, val_dataset = prepare_graph_ncf(ranking=False, use_features=use_features)

    print(model)

    # log
    now = datetime.now()
    hyperparams = {k: (', '.join([str(i) for i in v]) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    hyperparams['features_used'] = use_features
    model_name = f"{type(model).__name__}_{'with_features' if use_features or isinstance(model, AttentionNCF) else 'onehot'}"

    # init weights & biases
    wandb.init(project='DeepRecommendation',
               entity='michaelbzms',
               name=model_name + '_' + now.strftime(now.strftime("%d_%m_%Y_%H_%M")),      # run name
               group=model_name,    # group name --> hyperparameter tuning on group
               dir='../',
               config={
                   "learning_rate": lr,
                   "batch_size": batch_size,
                   "weight_decay": weight_decay,
                   **hyperparams
               })
    print(wandb.config)

    # train and save result
    monitored_metrics = train_model(model, train_dataset=training_dataset, val_dataset=val_dataset,
                                    lr=lr, weight_decay=weight_decay, batch_size=batch_size, val_batch_size=val_batch_size,
                                    early_stop=early_stop, final_model_path=final_model_path, checkpoint_model_path=checkpoint_model_path,
                                    max_epochs=max_epochs, patience=patience, stop_with_train_loss_instead=stop_with_train_loss_instead,
                                    wandb=wandb)

    # plot and save losses
    plot_train_val_losses(monitored_metrics['train_loss'], monitored_metrics['val_loss'])
