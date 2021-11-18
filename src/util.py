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
