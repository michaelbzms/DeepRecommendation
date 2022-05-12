from datetime import datetime
import wandb
from pathlib import Path

from content_providers.dynamic_profiles_provider import DynamicProfilesProvider
from content_providers.fixed_profiles_provider import FixedProfilesProvider, FixedItemProfilesOnlyProvider
from content_providers.graph_providers import OneHotGraphProvider, ProfilesGraphProvider
from content_providers.one_hot_provider import OneHotProvider
from neural_collaborative_filtering.datasets.dynamic_datasets import DynamicPointwiseDataset, DynamicRankingDataset
from neural_collaborative_filtering.datasets.fixed_datasets import FixedPointwiseDataset, FixedRankingDataset
from neural_collaborative_filtering.datasets.gnn_datasets import GraphPointwiseDataset, GraphRankingDataset
from neural_collaborative_filtering.eval import eval_model
from neural_collaborative_filtering.models.advanced_ncf import AttentionNCF
from neural_collaborative_filtering.models.basic_ncf import BasicNCF
from neural_collaborative_filtering.models.gnn_ncf import NGCF
from neural_collaborative_filtering.plots import plot_train_val_losses
from neural_collaborative_filtering.train import train_model
from globals import train_set_file, val_set_file, weight_decay, lr, batch_size, max_epochs, early_stop, \
    checkpoint_model_path, patience, dropout_rate, final_model_path, \
    val_batch_size, ranking_train_set_file, \
    ranking_val_set_file, test_set_file


def prepare_fixedinput_ncf(ranking=False, use_features=False, onehot_users=False, model_kwargs=None):
    """
    Set up the model and datasets to train an NCF model on fixed input.
    """
    if use_features:
        # content provider
        cp = FixedItemProfilesOnlyProvider() if onehot_users else FixedProfilesProvider()
        # model
        if model_kwargs is not None:
            model = BasicNCF(item_dim=cp.get_item_feature_dim(),
                             user_dim=cp.get_num_users() if onehot_users else cp.get_item_feature_dim(),
                             **model_kwargs)
        else:
            model = BasicNCF(item_dim=cp.get_item_feature_dim(),
                             user_dim=cp.get_num_users() if onehot_users else cp.get_item_feature_dim(),
                             item_emb=128, user_emb=128,
                             mlp_dense_layers=[256],
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
                             mlp_dense_layers=[256],
                             dropout_rate=dropout_rate)
    # datasets
    val_dataset = FixedPointwiseDataset(val_set_file, content_provider=cp)
    test_dataset = FixedPointwiseDataset(test_set_file, content_provider=cp)
    if ranking:
        training_dataset = FixedRankingDataset(ranking_train_set_file, content_provider=cp)
    else:
        training_dataset = FixedPointwiseDataset(train_set_file, content_provider=cp)

    return model, training_dataset, val_dataset, test_dataset


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
    val_dataset = DynamicPointwiseDataset(val_set_file, dynamic_provider=dpp)
    test_dataset = DynamicPointwiseDataset(test_set_file, dynamic_provider=dpp)
    if ranking:
        training_dataset = DynamicRankingDataset(ranking_train_set_file, dynamic_provider=dpp)
    else:
        training_dataset = DynamicPointwiseDataset(train_set_file, dynamic_provider=dpp)

    return model, training_dataset, val_dataset, test_dataset


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
                     mlp_dense_layers=[256],
                     dropout_rate=dropout_rate,
                     message_dropout=0.1)
    # datasets
    val_dataset = GraphPointwiseDataset(val_set_file, graph_content_provider=gcp)
    test_dataset = GraphPointwiseDataset(test_set_file, graph_content_provider=gcp)    # TODO: add val edges to gcp for this?
    if ranking:
        training_dataset = GraphRankingDataset(ranking_train_set_file, graph_content_provider=gcp)
    else:
        training_dataset = GraphPointwiseDataset(train_set_file, graph_content_provider=gcp)

    return model, training_dataset, val_dataset, test_dataset


def run_experiment(model, *, hparams, training_dataset, val_dataset, test_dataset=None,
                   use_features=False, ranking=False, onehot_users=False, save_model=True,
                   final_model_save_path=None, group_name='runs', project_name='DeepRecommendation', **kwargs):
    print('___________________ Running experiment ___________________')
    print(model)

    # log
    now = datetime.now()
    timestamp = now.strftime(now.strftime("%d_%m_%Y_%H_%M"))
    model_hyperparams = {k: (', '.join([str(i) for i in v]) if isinstance(v, list) else v) for k, v in model.kwargs.items()}
    model_hyperparams['ranking'] = ranking
    model_hyperparams['features_used'] = use_features
    model_name = f"{type(model).__name__}_{('item_features_but_users_onehot' if onehot_users else 'with_features') if use_features or isinstance(model, AttentionNCF) else 'onehot'}" \
                 + ('_ranking' if ranking else '')

    # if and where to save the trained model
    if save_model:
        if final_model_save_path is None:
            model_save_path = f'../models/{group_name}/{model_name}.pt'
            Path(f'../models/{group_name}').mkdir(parents=True, exist_ok=True)     # create dir if it does not exist
        else:
            model_save_path = final_model_save_path
    else:
        model_save_path = None

    if isinstance(model, AttentionNCF):
        pass

    # init weights & biases
    run = wandb.init(project=project_name,
                     entity='michaelbzms',
                     reinit=True,
                     name=model_name + '_' + timestamp,  # run name
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
                                    lr=hparams['lr'], weight_decay=hparams['weight_decay'],
                                    batch_size=hparams['batch_size'],
                                    val_batch_size=val_batch_size,      # not important
                                    early_stop=True, final_model_path=model_save_path,
                                    checkpoint_model_path=checkpoint_model_path,
                                    max_epochs=max_epochs, patience=patience,
                                    wandb=wandb)

    if test_dataset is not None:
        eval_model(model, test_dataset, val_batch_size, wandb=wandb, ranking=ranking, doplots=False)

    run.finish()

    return monitored_metrics


if __name__ == '__main__':
    use_features = True
    ranking = True
    onehot_users = False

    # prepare model, train and val datasets (Pointwise val dataset always needed for NDCG eval)
    model, training_dataset, val_dataset, test_dataset = prepare_fixedinput_ncf(ranking=ranking, use_features=use_features, onehot_users=onehot_users)
    # model, training_dataset, val_dataset, test_dataset = prepare_attention_ncf(ranking=ranking)
    # model, training_dataset, val_dataset, test_dataset = prepare_graph_ncf(ranking=ranking, use_features=use_features)

    print(model)

    # train and save result
    # monitored_metrics = run_experiment(model,
    #                                    hparams={
    #                                        'lr': lr,
    #                                        'batch_size': batch_size,
    #                                        'weight_decay': weight_decay
    #                                    },
    #                                    training_dataset=training_dataset,
    #                                    val_dataset=val_dataset,
    #                                    pointwise_val_dataset=pointwise_val_dataset,
    #                                    final_model_save_path=final_model_path,
    #                                    use_features=use_features,
    #                                    ranking=ranking)

    # train and save result at `final_model_save_path`
    monitored_metrics = train_model(model, train_dataset=training_dataset, val_dataset=val_dataset,
                                    lr=3e-4, weight_decay=1e-5,
                                    batch_size=1024,
                                    val_batch_size=val_batch_size,  # not important
                                    early_stop=True, final_model_path=final_model_path,
                                    checkpoint_model_path=checkpoint_model_path,
                                    max_epochs=max_epochs, patience=patience,
                                    wandb=None)

    # plot and save losses
    plot_train_val_losses(monitored_metrics['train_loss'], monitored_metrics['val_ndcg'] if ranking else monitored_metrics['val_loss'])
