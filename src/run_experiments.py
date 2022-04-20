from train_model import prepare_attention_ncf, prepare_graph_ncf, prepare_fixedinput_ncf, run_experiment


if __name__ == '__main__':
    #############################
    # define experiments to run #
    #############################
    fixed_experiments = [
        {'use_features': False,
         'ranking': False,
         'model_kwargs': {
             'item_emb': 128, 'user_emb': 128,
             'mlp_dense_layers': [256],
             'dropout_rate': 0.2
         },
         'lr': 3e-4,
         'batch_size': 256,
         'weight_decay': 1e-5,
         },
        {'use_features': False,
         'ranking': False,
         'model_kwargs': {
            'item_emb': 128, 'user_emb': 128,
            'mlp_dense_layers': [256, 128],
            'dropout_rate': 0.2
         },
         'lr': 3e-4,
         'batch_size': 256,
         'weight_decay': 1e-5,
         },
        {'use_features': False,
         'ranking': False,
         'model_kwargs': {
             'item_emb': 256, 'user_emb': 256,
             'mlp_dense_layers': [256, 128],
             'dropout_rate': 0.2
         },
         'lr': 3e-4,
         'batch_size': 256,
         'weight_decay': 1e-5
         }
    ]

    attention_experiments = [
        {'use_features': False,
         'ranking': False,
         'model_kwargs': {
             'item_emb': 128, 'user_emb': 128,
             'att_dense': None,
             'mlp_dense_layers': [256],
             'dropout_rate': 0.2
         },
         'lr': 3e-4,
         'batch_size': 256,
         'weight_decay': 1e-5
         },
        {'use_features': False,
         'ranking': False,
         'model_kwargs': {
             'item_emb': 128, 'user_emb': 128,
             'att_dense': 8,
             'mlp_dense_layers': [256],
             'dropout_rate': 0.2
         },
         'lr': 3e-4,
         'batch_size': 256,
         'weight_decay': 1e-5
         }
    ]

    graph_experiments = [
        {'use_features': False,
         'ranking': False,
         'model_kwargs': {
             'node_emb': 64,
             'gnn_dense_layers': [64, 64],
             'mlp_dense_layers': [128],
             'dropout_rate': 0.2,
             'gnn_dropout_rate': 0.1
         },
         'lr': 3e-4,
         'batch_size': 256,
         'weight_decay': 1e-5
         },
        {'use_features': False,
         'ranking': False,
         'model_kwargs': {
             'node_emb': 128,
             'gnn_dense_layers': [64, 64],
             'mlp_dense_layers': [128],
             'dropout_rate': 0.2,
             'gnn_dropout_rate': 0.1
         },
         'lr': 3e-4,
         'batch_size': 256,
         'weight_decay': 1e-5
         }
    ]

    for exp in fixed_experiments:
        # prepare experiment (model and datasets)
        model, training_dataset, val_dataset, pointwise_val_dataset = prepare_fixedinput_ncf(**exp)

        # run experiment
        run_experiment(model,
                       hparams={
                           'lr': exp['lr'],
                           'batch_size': exp['batch_size'],
                           'weight_decay': exp['weight_decay']
                       },
                       training_dataset=training_dataset,
                       val_dataset=val_dataset,
                       pointwise_val_dataset=pointwise_val_dataset,
                       final_model_save_path=None,
                       use_features=exp['use_features'],
                       ranking=exp['ranking'])

    for exp in attention_experiments:
        # prepare experiment (model and datasets)
        model, training_dataset, val_dataset, pointwise_val_dataset = prepare_attention_ncf(**exp)

        # run experiment
        run_experiment(model,
                       hparams={
                           'lr': exp['lr'],
                           'batch_size': exp['batch_size'],
                           'weight_decay': exp['weight_decay']
                       },
                       training_dataset=training_dataset,
                       val_dataset=val_dataset,
                       pointwise_val_dataset=pointwise_val_dataset,
                       final_model_save_path=None,
                       use_features=True,           # only works with features
                       ranking=exp['ranking'])

    for exp in graph_experiments:
        # prepare experiment (model and datasets)
        model, training_dataset, val_dataset, pointwise_val_dataset = prepare_graph_ncf(**exp)

        # run experiment
        run_experiment(model,
                       hparams={
                           'lr': exp['lr'],
                           'batch_size': exp['batch_size'],
                           'weight_decay': exp['weight_decay']
                       },
                       training_dataset=training_dataset,
                       val_dataset=val_dataset,
                       pointwise_val_dataset=pointwise_val_dataset,
                       final_model_save_path=None,
                       use_features=exp['use_features'],
                       ranking=exp['ranking'])
