import torch
from torch import nn


def build_MLP_layers(input_size, layer_sizes: list, dropout_rate, output_size=1) -> nn.Sequential:
    """
    Build a chain of Dense layers of specified sizes with ReLU as the activation function
    and a dropout layer if dropout_rate is not None.
    """
    mlp_dense_layers = list(layer_sizes)     # make mutable if it's not
    mlp_dense_layers.append(output_size)     # the output layer
    num_layers = len(mlp_dense_layers)
    mlp_layers = [nn.Linear(input_size, mlp_dense_layers[0])]  # input layer
    for i in range(1, num_layers):
        mlp_layers.append(nn.ReLU())
        if dropout_rate is not None: mlp_layers.append(nn.Dropout(dropout_rate))
        mlp_layers.append(nn.Linear(mlp_dense_layers[i-1], mlp_dense_layers[i]))
    return nn.Sequential(*mlp_layers)


def load_model(file, ModelClass=None, **kargs):
    """
    Loads a model of the given class that is saved on given file.
    Expects a dictionary with saved model's hyperparameters to be there as well.
    Optionally passes any extra arguments it was called with to the model's constructor.
    """
    state, kwargs = torch.load(file)
    kwargs = dict(kwargs, **kargs)          # add any extra args passed which might not have been saved in file
    if ModelClass is None:
        return state, kwargs
    else:
        model = ModelClass(**kwargs)
        model.load_state_dict(state)
        return model
