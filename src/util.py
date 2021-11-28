import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from torch import nn


def one_hot_encode(actual_value, ordered_possible_values: np.array) -> np.array:
    return np.array(ordered_possible_values == actual_value, dtype=np.float64)


def multi_hot_encode(actual_values: set, ordered_possible_values: list) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform([actual_values])
    return binary_format


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
