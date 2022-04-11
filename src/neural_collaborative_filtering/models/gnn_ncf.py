import torch
import torch_scatter
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from neural_collaborative_filtering.models.base import GNN_NCF
from neural_collaborative_filtering.util import build_MLP_layers


def pass_gnn_layers(gnn_convs, graph, **kwargs):
    # encode with GNN
    graph_emb = graph.x  # initial representation
    hs = []
    for gnn_conv in gnn_convs:
        graph_emb = gnn_conv(graph_emb, graph.edge_index, **kwargs)
        graph_emb = F.leaky_relu(graph_emb)
        hs.append(graph_emb)
    # concat all intermediate representations
    combined_graph_emb = torch.cat(hs, dim=1)
    return combined_graph_emb


# class NGCFConv(MessagePassing):
#     def __init__(self):
#         super(NGCFConv, self).__init__()
#
#     def message(self, x_j, x_i, norm):
#         return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))
#
#     def aggregate(self, x, messages, index):
#         out = torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
#         return out
#
#     def forward(self, x, edge_index):
#         norm = compute_normalization(x, edge_index)
#         out = self.propagate(edge_index, x=(x, x), norm=norm)  # Update step
#         out += self.lin_1(x)
#         out = F.dropout(out, self.dropout, self.training)
#         return F.leaky_relu(out)


class NGCFConv(MessagePassing):
    """
    NGCF Conv layer implementation taken and modified from:
    https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377
    """
    def __init__(self, latent_dim, dropout, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr='add', **kwargs)
        self.dropout = dropout
        self.W1 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.W2 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages
        out = self.propagate(edge_index, x=(x, x), norm=norm)

        # add self-message
        out += self.W1(x)

        # TODO: Is this message dropout??? --> I don't think it is...
        out = F.dropout(out, self.dropout, self.training)

        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.W1(x_j) + self.W2(x_j * x_i))

    # Note: this is probably not needed because of aggr='add' in __init__()
    # def aggregate(self, x, messages, index):
    #     out = torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
    #     return out


class GCN_NCF(GNN_NCF):
    def __init__(self, gnn_hidden_layers=None, item_emb=128, user_emb=128, mlp_dense_layers=None, dropout_rate=0.2):
        super(GCN_NCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]  # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [128, 64, 64]  # default
        self.kwargs = {'gnn_hidden_layers': gnn_hidden_layers,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers}

        self.gnn_convs = nn.ModuleList(
            [GCNConv(in_channels=-1 if i == 0 else gnn_hidden_layers[i - 1],
                     out_channels=gnn_hidden_layers[i],
                     normalize=False,
                     improved=True) for i in range(len(gnn_hidden_layers))]
        )

        self.item_embeddings = nn.Sequential(
            nn.Linear(gnn_hidden_layers[-1], item_emb),
            nn.ReLU()
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(gnn_hidden_layers[-1], user_emb),
            nn.ReLU()
        )
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

        # self.MLP = build_MLP_layers(gnn_hidden_layers[-1] * 2, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, GNN_Dataset)

    def forward(self, *, graph, userIds, itemIds):
        # encode all graph nodes with GNN
        combined_graph_emb = pass_gnn_layers(self.gnn_convs, graph, edge_weight=graph.edge_attr)

        # find embeddings of items in batch
        item_emb = combined_graph_emb[itemIds.long()]

        # find embeddings of users in batch
        user_emb = combined_graph_emb[userIds.long()]

        # use these to forward the NCF model
        item_emb = self.item_embeddings(item_emb)
        user_emb = self.user_embeddings(user_emb)
        combined = torch.cat((item_emb, user_emb), dim=1)
        out = self.MLP(combined)

        return out


class GAT_NCF(GNN_NCF):
    def __init__(self, initial_repr_dim=-1, gnn_hidden_layers=None, item_emb=128, user_emb=128,
                 mlp_dense_layers=None, num_heads=1, extra_emb_layers=False,
                 dropout_rate=0.2, edge_dim=-1):
        super(GAT_NCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]  # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [128]  # default
        self.kwargs = {'gnn_hidden_layers': gnn_hidden_layers,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers,
                       'num_heads': num_heads,
                       'extra_emb_layers': extra_emb_layers,
                       'initial_repr_dim': initial_repr_dim,
                       'edge_dim': edge_dim}

        self.gnn_convs = nn.ModuleList(
            [GATConv(in_channels=initial_repr_dim if i == 0 else gnn_hidden_layers[i - 1],
                     out_channels=gnn_hidden_layers[i],
                     edge_dim=edge_dim,
                     heads=num_heads) for i in range(len(gnn_hidden_layers))]
        )

        self.extra_emb_layers = extra_emb_layers

        if extra_emb_layers:
            self.item_embeddings = nn.Sequential(
                nn.Linear(sum(gnn_hidden_layers) * num_heads, item_emb),
                nn.ReLU()
            )
            self.user_embeddings = nn.Sequential(
                nn.Linear(sum(gnn_hidden_layers) * num_heads, user_emb),
                nn.ReLU()
            )
            self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)
        else:
            self.MLP = build_MLP_layers(sum(gnn_hidden_layers) * num_heads * 2, mlp_dense_layers,
                                        dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, GNN_Dataset)

    def forward(self, *, graph, userIds, itemIds):  # needs to be True for training only I think
        # encode all graph nodes with GNN
        combined_graph_emb = pass_gnn_layers(self.gnn_convs, graph, edge_attr=graph.edge_attr)

        # find embeddings of items in batch
        item_emb = combined_graph_emb[itemIds.long()]

        # find embeddings of users in batch
        user_emb = combined_graph_emb[userIds.long()]

        # use these to forward the NCF model
        if self.extra_emb_layers:
            item_emb = self.item_embeddings(item_emb)
            user_emb = self.user_embeddings(user_emb)
        combined = torch.cat((item_emb, user_emb), dim=1)
        out = self.MLP(combined)

        return out
