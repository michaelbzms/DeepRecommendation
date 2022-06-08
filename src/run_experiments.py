from train_model import prepare_attention_ncf, prepare_graph_ncf, prepare_fixedinput_ncf, run_experiment


project_name = '5kusers1kmovies_Ranking'
group_name = 'runs'
save_models = True
eval_on_test_also = True


if __name__ == '__main__':
    #############################
    # define experiments to run #
    #############################
    fixed_experiments = [
        # {'use_features': True,
        #  'use_ranking': True,
        #  'onehot_users': False,
        #  'model_kwargs': {
        #      'item_emb': 128, 'user_emb': 128,
        #      'mlp_dense_layers': [256],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 7e-4,
        #  'batch_size': 512,
        #  'weight_decay': 1e-5,
        #  },
        # {'use_features': True,
        #  'use_ranking': True,
        #  'onehot_users': False,
        #  'model_kwargs': {
        #      'item_emb': 128, 'user_emb': 128,
        #      'mlp_dense_layers': [256],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 7e-4,
        #  'batch_size': 512,
        #  'weight_decay': 1e-5,
        #  },
        # {'use_features': True,
        #  'use_ranking': True,
        #  'onehot_users': False,
        #  'model_kwargs': {
        #      'item_emb': 128, 'user_emb': 128,
        #      'mlp_dense_layers': [256],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 5e-4,
        #  'batch_size': 512,
        #  'weight_decay': 1e-5,
        #  },
        # {'use_features': True,
        #  'use_ranking': False,
        #  'model_kwargs': {
        #      'item_emb': 128, 'user_emb': 128,
        #      'mlp_dense_layers': [256],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 1e-3,
        #  'batch_size': 128,
        #  'weight_decay': 1e-5
        #  },
        # {'use_features': False,
        #  'use_ranking': False,
        #  'model_kwargs': {
        #      'item_emb': 256, 'user_emb': 256,
        #      'mlp_dense_layers': [512, 256],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 1e-3,
        #  'batch_size': 64,
        #  'weight_decay': 1e-5
        #  },
        # {'use_features': True,
        #  'use_ranking': False,
        #  'model_kwargs': {
        #      'item_emb': 128, 'user_emb': 128,
        #      'mlp_dense_layers': [256],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 1e-3,
        #  'batch_size': 64,
        #  'weight_decay': 1e-5,
        #  },
        # {'use_features': True,
        #  'use_ranking': False,
        #  'model_kwargs': {
        #      'item_emb': 256, 'user_emb': 256,
        #      'mlp_dense_layers': [512],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 1e-3,
        #  'batch_size': 64,
        #  'weight_decay': 1e-5
        #  },
        # {'use_features': True,
        #  'use_ranking': False,
        #  'model_kwargs': {
        #      'item_emb': 256, 'user_emb': 256,
        #      'mlp_dense_layers': [512, 256],
        #      'dropout_rate': 0.2
        #  },
        #  'lr': 1e-3,
        #  'batch_size': 64,
        #  'weight_decay': 1e-5
        #  }
    ]

    attention_experiments = [
        # {'use_features': True,
        #  'use_ranking': False,
        #  'model_kwargs': {
        #      'item_emb': 128, 'user_emb': 128,
        #      'att_dense': 128,
        #      'use_cos_sim_instead': False,
        #      'mlp_dense_layers': [256],
        #      'dropout_rate': 0.2,
        #      'message_dropout': 0.1
        #  },
        #  'lr': 1e-3,
        #  'batch_size': 128,
        #  'weight_decay': 1e-5
        #  }
    ]

    graph_experiments = [
        {'use_features': True,
         'use_ranking': False,
         'model_kwargs': {
             'node_emb': 64,
             'num_gnn_layers': 2,
             'mlp_dense_layers': [128],
             'dropout_rate': 0.2,
             'message_dropout': 0.1,  # TODO? -> reduces memory requirements
             'concat': False
         },
         'lr': 1e-3,
         'batch_size': 512,
         'weight_decay': 1e-5
         },
    ]

    for exp in fixed_experiments:
        # prepare experiment (model and datasets)
        model, training_dataset, val_dataset, test_dataset = prepare_fixedinput_ncf(
            use_features=exp['use_features'],
            onehot_users=exp['onehot_users'] if 'onehot_users' in exp else False,
            ranking=exp['use_ranking'],
            model_kwargs=exp['model_kwargs']
        )

        # run experiment
        run_experiment(model,
                       hparams={
                           'lr': exp['lr'],
                           'batch_size': exp['batch_size'],
                           'weight_decay': exp['weight_decay']
                       },
                       training_dataset=training_dataset,
                       val_dataset=val_dataset,
                       test_dataset=test_dataset if eval_on_test_also else None,
                       onehot_users=exp['onehot_users'] if 'onehot_users' in exp else False,
                       save_model=save_models,
                       use_features=exp['use_features'],
                       ranking=exp['use_ranking'],
                       group_name=group_name,
                       project_name=project_name)

    for exp in attention_experiments:
        # prepare experiment (model and datasets)
        model, training_dataset, val_dataset, test_dataset = prepare_attention_ncf(
            ranking=exp['use_ranking'],
            model_kwargs=exp['model_kwargs']
        )

        # run experiment
        run_experiment(model,
                       hparams={
                           'lr': exp['lr'],
                           'batch_size': exp['batch_size'],
                           'weight_decay': exp['weight_decay']
                       },
                       training_dataset=training_dataset,
                       val_dataset=val_dataset,
                       test_dataset=test_dataset if eval_on_test_also else None,
                       save_model=save_models,
                       use_features=True,           # only works with features
                       ranking=exp['use_ranking'],
                       group_name=group_name,
                       project_name=project_name)

    for exp in graph_experiments:
        # prepare experiment (model and datasets)
        model, training_dataset, val_dataset, test_dataset = prepare_graph_ncf(
            use_features=exp['use_features'],
            ranking=exp['use_ranking'],
            model_kwargs=exp['model_kwargs']
        )

        # run experiment
        run_experiment(model,
                       hparams={
                           'lr': exp['lr'],
                           'batch_size': exp['batch_size'],
                           'weight_decay': exp['weight_decay']
                       },
                       training_dataset=training_dataset,
                       val_dataset=val_dataset,
                       test_dataset=test_dataset if eval_on_test_also else None,
                       save_model=save_models,
                       use_features=exp['use_features'],
                       ranking=exp['use_ranking'],
                       group_name=group_name,
                       project_name=project_name)
