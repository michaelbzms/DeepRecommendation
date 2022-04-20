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


def prepare_fixedinput_ncf(ranking=False, use_features=False, model_kwargs=None):
    """
    Set up the model and datasets to train an NCF model on fixed input.
    """
    if use_features:
        # content provider
        cp = FixedProfilesProvider()
        # model
        if model_kwargs is not None:
            model = BasicNCF(item_dim=cp.get_item_feature_dim(),
                             user_dim=cp.get_item_feature_dim(),
                             **model_kwargs)
        else:
            model = BasicNCF(item_dim=cp.get_item_feature_dim(),
                             user_dim=cp.get_item_feature_dim(),
                             item_emb=128, user_emb=128,
                             mlp_dense_layers=[256, 128],
                             dropout_rate=dropout_rate)
    else:
        # content provider
        cp = OneHotProvider()
        # model
        if model_kwargs is not None:
            model = BasicNCF(item_dim=cp.get_num_items(),
                             user_dim=cp.get_num_users(),
                             **model_kwargs)
        else:
            model = BasicNCF(item_dim=cp.get_num_items(),
                             user_dim=cp.get_num_users(),
                             item_emb=128, user_emb=128,
                             mlp_dense_layers=[256, 128],
                             dropout_rate=dropout_rate)
    # datasets
    pointwise_val_dataset = FixedPointwiseDataset(val_set_file, content_provider=cp)
    if ranking:
        training_dataset = FixedRankingDataset(ranking_train_set_file, content_provider=cp)
        val_dataset = FixedRankingDataset(ranking_val_set_file, content_provider=cp)
    else:
        training_dataset = FixedPointwiseDataset(train_set_file, content_provider=cp)
        val_dataset = pointwise_val_dataset

    return model, training_dataset, val_dataset, pointwise_val_dataset


def prepare_attention_ncf(ranking=False, model_kwargs=None):
    """
    Set up the model and datasets to train the attention NCF model on dynamic input
    (i.e. dynamic user profiles).
    """
    # content provider
    dpp = DynamicProfilesProvider()
    # model
    if model_kwargs is not None:
        model = AttentionNCF(dpp.get_item_feature_dim(), **model_kwargs)
    else:
        model = AttentionNCF(dpp.get_item_feature_dim(),
                             item_emb=128, user_emb=128,
                             att_dense=None,
                             mlp_dense_layers=[256],
                             dropout_rate=dropout_rate)

    # datasets
    pointwise_val_dataset = DynamicPointwiseDataset(val_set_file, dynamic_provider=dpp)
    if ranking:
        training_dataset = DynamicRankingDataset(ranking_train_set_file, dynamic_provider=dpp)
        val_dataset = DynamicRankingDataset(ranking_val_set_file, dynamic_provider=dpp)
    else:
        training_dataset = DynamicPointwiseDataset(train_set_file, dynamic_provider=dpp)
        val_dataset = pointwise_val_dataset

    return model, training_dataset, val_dataset, pointwise_val_dataset


def prepare_graph_ncf(ranking=False, use_features=False, model_kwargs=None):
    """
    Set up the model and datasets to train a graph NCF model on fixed input.
    """

    if use_features:
        # content provider
        gcp = ProfilesGraphProvider(train_set_file)
    else:
        # content provider
        gcp = OneHotGraphProvider(train_set_file)

    # model
    if model_kwargs is not None:
        model = NGCF(item_dim=gcp.get_item_dim(),
                     user_dim=gcp.get_user_dim(),
                     **model_kwargs)
    else:
        model = NGCF(item_dim=gcp.get_item_dim(),
                     user_dim=gcp.get_user_dim(),
                     gnn_hidden_layers=[64, 64],
                     node_emb=64,
                     mlp_dense_layers=[256, 128],
                     dropout_rate=dropout_rate,
                     gnn_dropout_rate=0.1)
    # datasets
    pointwise_val_dataset = GraphPointwiseDataset(val_set_file, graph_content_provider=gcp)
    if ranking:
        training_dataset = GraphRankingDataset(ranking_train_set_file, graph_content_provider=gcp)
        val_dataset = GraphRankingDataset(ranking_val_set_file, graph_content_provider=gcp)
    else:
        training_dataset = GraphPointwiseDataset(train_set_file, graph_content_provider=gcp)
        val_dataset = pointwise_val_dataset

    return model, training_dataset, val_dataset, pointwise_val_dataset


def run_experiment(model, *, hparams, training_dataset, val_dataset, pointwise_val_dataset,
                   use_features=False, ranking=False, final_model_save_path=None, **kwargs):
    print('___________________ Running experiment ___________________')
    print(model)

    # log
    now = datetime.now()
    model_hyperparams = {k: (', '.join([str(i) for i in v]) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    model_hyperparams['ranking'] = ranking
    model_hyperparams['features_used'] = use_features
    model_name = f"{type(model).__name__}_{'with_features' if use_features or isinstance(model, AttentionNCF) else 'onehot'}"

    # init weights & biases
    run = wandb.init(project='DeepRecommendation',
                     entity='michaelbzms',
                     reinit=True,
                     name=model_name + '_' + now.strftime(now.strftime("%d_%m_%Y_%H_%M")),  # run name
                     group=model_name,  # group name --> hyperparameter tuning on group
                     dir='../',
                     config={
                         "learning_rate": hparams['lr'],
                         "batch_size": hparams['batch_size'],
                         "weight_decay": hparams['weight_decay'],
                         **model_hyperparams
                     })
    print(wandb.config)

    # train and save result at `final_model_save_path`
    monitored_metrics = train_model(model, train_dataset=training_dataset, val_dataset=val_dataset,
                                    pointwise_val_dataset=pointwise_val_dataset,
                                    lr=hparams['lr'], weight_decay=hparams['weight_decay'],
                                    batch_size=hparams['batch_size'],
                                    val_batch_size=val_batch_size,      # not important
                                    early_stop=True, final_model_path=final_model_save_path,
                                    checkpoint_model_path=checkpoint_model_path,
                                    max_epochs=max_epochs, patience=patience,
                                    wandb=wandb)

    run.finish()

    return monitored_metrics


if __name__ == '__main__':
    use_features = False
    ranking = False

    # prepare model, train and val datasets (Pointwise val dataset always needed for NDCG eval)
    model, training_dataset, val_dataset, pointwise_val_dataset = prepare_fixedinput_ncf(ranking=ranking, use_features=use_features)
    # model, training_dataset, val_dataset = prepare_attention_ncf(ranking=ranking)
    # model, training_dataset, val_dataset = prepare_graph_ncf(ranking=ranking, use_features=use_features)

    # train and save result
    monitored_metrics = run_experiment(model,
                                       hparams={
                                           'lr': lr,
                                           'batch_size': batch_size,
                                           'weight_decay': weight_decay
                                       },
                                       training_dataset=training_dataset,
                                       val_dataset=val_dataset,
                                       pointwise_val_dataset=pointwise_val_dataset,
                                       final_model_save_path=final_model_path,
                                       use_features=use_features,
                                       ranking=ranking)

    # plot and save losses
    plot_train_val_losses(monitored_metrics['train_loss'], monitored_metrics['val_loss'])
