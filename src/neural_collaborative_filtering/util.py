import torch
from torch import nn


def build_MLP_layers(input_size, layer_sizes: list, dropout_rate) -> nn.Sequential:
    mlp_dense_layers = list(layer_sizes)  # make mutable if it's not
    mlp_dense_layers.append(1)            # the output layer
    num_layers = len(mlp_dense_layers)
    mlp_layers = [nn.Linear(input_size, mlp_dense_layers[0])]  # input layer
    for i in range(1, num_layers):
        mlp_layers.append(nn.ReLU())
        if dropout_rate is not None: mlp_layers.append(nn.Dropout(dropout_rate))
        mlp_layers.append(nn.Linear(mlp_dense_layers[i-1], mlp_dense_layers[i]))
    return nn.Sequential(*mlp_layers)


def load_model_state_and_params(file, ModelClass=None, **kargs):
    state, kwargs = torch.load(file)
    kwargs = dict(kwargs, **kargs)        # add any extra args passed which might not have been saved in file
    if ModelClass is None:
        return state, kwargs
    else:
        model = ModelClass(**kwargs)
        model.load_state_dict(state)
        return model


def load_model(file, ModelClass, **kargs):
    return load_model_state_and_params(file, ModelClass, **kargs)
