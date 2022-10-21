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
from neural_collaborative_filtering.models.attention_ncf import AttentionNCF
from neural_collaborative_filtering.models.basic_ncf import BasicNCF
from neural_collaborative_filtering.models.gnn_ncf import GraphNCF
from neural_collaborative_filtering.models.mf import MF
from neural_collaborative_filtering.plots import plot_train_val_losses
from neural_collaborative_filtering.train import train_model
from globals import train_set_file, val_set_file, max_epochs, \
    checkpoint_model_path, patience, final_model_path, \
    val_batch_size, ranking_train_set_file, \
    test_set_file, train_and_val_set_file, num_workers


def prepare_fixedinput_ncf(ranking=False, use_features=False, onehot_users=False, model_kwargs=None, include_val_ratings_to_user_profiles=False):
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
                             dropout_rate=0.2)
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
                             dropout_rate=0.2)
    # datasets
    val_dataset = FixedPointwiseDataset(val_set_file, content_provider=cp)
    test_dataset = FixedPointwiseDataset(test_set_file, content_provider=cp)
    if ranking:
        training_dataset = FixedRankingDataset(ranking_train_set_file, content_provider=cp)
    else:
        training_dataset = FixedPointwiseDataset(train_set_file, content_provider=cp)

    if include_val_ratings_to_user_profiles and use_features and not onehot_users:
        cp_with_val = FixedProfilesProvider(include_val_ratings_to_user_profiles=True)
        test_dataset_with_val = FixedPointwiseDataset(test_set_file, content_provider=cp_with_val)
    else:
        test_dataset_with_val = None

    return model, training_dataset, val_dataset, test_dataset, test_dataset_with_val


def prepare_matrix_factorization(use_features=False, onehot_users=False, model_kwargs=None):
    """
    Set up the model and datasets for purelly CF method of MF.
    """
    if use_features:
        # content provider
        cp = FixedItemProfilesOnlyProvider() if onehot_users else FixedProfilesProvider()
        # model
        if model_kwargs is not None:
            model = MF(item_dim=cp.get_item_feature_dim(),
                       user_dim=cp.get_num_users() if onehot_users else cp.get_item_feature_dim(),
                       **model_kwargs)
        else:
            model = MF(item_dim=cp.get_item_feature_dim(),
                       user_dim=cp.get_num_users() if onehot_users else cp.get_item_feature_dim(),
                       item_emb=128, user_emb=128)
    else:
        # content provider
        cp = OneHotProvider()
        # model
        if model_kwargs is not None:
            model = MF(item_dim=cp.get_num_items(),
                       user_dim=cp.get_num_users(),
                       **model_kwargs)
        else:
            model = MF(item_dim=cp.get_num_items(),
                       user_dim=cp.get_num_users(),
                       item_emb=128, user_emb=128)
    # datasets
    training_dataset = FixedPointwiseDataset(train_set_file, content_provider=cp)
    val_dataset = FixedPointwiseDataset(val_set_file, content_provider=cp)
    test_dataset = FixedPointwiseDataset(test_set_file, content_provider=cp)
    test_dataset_with_val = None

    return model, training_dataset, val_dataset, test_dataset, test_dataset_with_val


def prepare_attention_ncf(ranking=False, model_kwargs=None, include_val_ratings_to_user_profiles=False):
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
                             att_dense=128,
                             use_cos_sim_instead=False,
                             mlp_dense_layers=[256],
                             message_dropout=None,
                             dropout_rate=0.2)

    # datasets
    val_dataset = DynamicPointwiseDataset(val_set_file, dynamic_provider=dpp)
    test_dataset = DynamicPointwiseDataset(test_set_file, dynamic_provider=dpp)
    if ranking:
        training_dataset = DynamicRankingDataset(ranking_train_set_file, dynamic_provider=dpp)
    else:
        training_dataset = DynamicPointwiseDataset(train_set_file, dynamic_provider=dpp)

    if include_val_ratings_to_user_profiles:
        dpp_with_val = DynamicProfilesProvider(include_val_ratings_to_user_profiles=True)
        test_dataset_with_val = DynamicPointwiseDataset(test_set_file, dynamic_provider=dpp_with_val)
    else:
        test_dataset_with_val = None

    return model, training_dataset, val_dataset, test_dataset, test_dataset_with_val


def prepare_graph_ncf(ranking=False, use_features=False, hetero=True, binary=False, model_kwargs=None, include_val_ratings_to_user_profiles=False):
    """
    Set up the model and datasets to train a graph NCF model on fixed input.
    """

    if use_features:
        # content provider
        gcp = ProfilesGraphProvider(train_set_file, binary=binary)
    else:
        # content provider
        gcp = OneHotGraphProvider(train_set_file, binary=binary)

    # model
    if model_kwargs is not None:
        model = GraphNCF(item_dim=gcp.get_item_dim(),
                         user_dim=gcp.get_user_dim(),
                         hetero=hetero,
                         **model_kwargs)
    else:
        model = GraphNCF(item_dim=gcp.get_item_dim(),
                         user_dim=gcp.get_user_dim(),
                         node_emb=64,
                         mlp_dense_layers=[128],
                         num_gnn_layers=3,
                         dropout_rate=0.2,
                         message_dropout=0.1,
                         use_dot_product=False,
                         hetero=hetero)
    # datasets
    val_dataset = GraphPointwiseDataset(val_set_file, graph_content_provider=gcp)
    test_dataset = GraphPointwiseDataset(test_set_file, graph_content_provider=gcp)
    if ranking:
        training_dataset = GraphRankingDataset(ranking_train_set_file, graph_content_provider=gcp)
    else:
        training_dataset = GraphPointwiseDataset(train_set_file, graph_content_provider=gcp)

    if include_val_ratings_to_user_profiles and use_features:
        gcp_with_val = ProfilesGraphProvider(train_and_val_set_file, binary=binary, include_val_ratings_to_user_profiles=True)
        test_dataset_with_val = GraphPointwiseDataset(test_set_file, graph_content_provider=gcp_with_val)
    else:
        test_dataset_with_val = None

    return model, training_dataset, val_dataset, test_dataset, test_dataset_with_val


def run_experiment(model, *, hparams, training_dataset, val_dataset, test_dataset=None, test_dataset_with_val=None,
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
    model_name = ('ranking_' if ranking else '') + f"{type(model).__name__}_{('item_features_but_users_onehot' if onehot_users else 'with_features') if use_features or isinstance(model, AttentionNCF) else 'onehot'}"

    # if and where to save the trained model
    if save_model:
        if final_model_save_path is None:
            model_save_path = f'../models/{group_name}/{model_name}{model.important_hypeparams()}.pt'
            Path(f'../models/{group_name}').mkdir(parents=True, exist_ok=True)     # create dir if it does not exist
        else:
            model_save_path = final_model_save_path
    else:
        model_save_path = None

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
    monitored_metrics = None
    try:
        monitored_metrics = train_model(model, train_dataset=training_dataset, val_dataset=val_dataset,
                                        lr=hparams['lr'], weight_decay=hparams['weight_decay'],
                                        batch_size=hparams['batch_size'],
                                        val_batch_size=val_batch_size,      # not important
                                        early_stop=True, final_model_path=model_save_path,
                                        checkpoint_model_path=checkpoint_model_path,
                                        max_epochs=max_epochs, patience=patience,
                                        wandb=wandb,
                                        num_workers=num_workers)

        if test_dataset is not None:
            eval_model(model, test_dataset, val_batch_size, wandb=wandb, ranking=ranking, doplots=False)
        if test_dataset_with_val is not None:
            eval_model(model, test_dataset_with_val, val_batch_size, wandb=wandb, ranking=ranking, doplots=False, val_ratings_included=True)
        run.finish()
    except:
        run.finish()

    return monitored_metrics


if __name__ == '__main__':
    # parameters
    use_features = True
    ranking = False
    onehot_users = False

    # prepare model, train and val datasets
    # model, training_dataset, val_dataset, test_dataset, test_dataset_with_val = prepare_matrix_factorization(use_features=use_features)
    # model, training_dataset, val_dataset, test_dataset, test_dataset_with_val = prepare_fixedinput_ncf(ranking=ranking, use_features=use_features, onehot_users=onehot_users)
    model, training_dataset, val_dataset, test_dataset, test_dataset_with_val = prepare_attention_ncf(ranking=ranking)
    # model, training_dataset, val_dataset, test_dataset, test_dataset_with_val = prepare_graph_ncf(ranking=ranking, use_features=use_features)

    # model to use
    print(model)

    # train and save result at `final_model_save_path`
    monitored_metrics = train_model(model, train_dataset=training_dataset, val_dataset=val_dataset,
                                    lr=1e-4, weight_decay=1e-5,
                                    batch_size=256,
                                    val_batch_size=val_batch_size,  # not important
                                    early_stop=True, final_model_path=final_model_path,
                                    checkpoint_model_path=checkpoint_model_path,
                                    max_epochs=max_epochs, patience=patience,
                                    wandb=None)

    # evaluate on test set
    if test_dataset is not None:
        eval_model(model, test_dataset, val_batch_size, wandb=None, ranking=ranking, doplots=False)
    if test_dataset_with_val is not None:
        eval_model(model, test_dataset_with_val, val_batch_size, wandb=None, ranking=ranking, doplots=False, val_ratings_included=True)

    # plot and save losses
    plot_train_val_losses(monitored_metrics['train_loss'], monitored_metrics['val_ndcg'] if ranking else monitored_metrics['val_loss'])
