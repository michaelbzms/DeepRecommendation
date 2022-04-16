import torch
import torch_scatter
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from neural_collaborative_filtering.datasets.gnn_datasets import GraphPointwiseDataset, GraphRankingDataset
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


class NGCFConv(MessagePassing):
    """
    NGCF Conv layer implementation taken and modified from:
    https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377
    """
    def __init__(self, in_channels, out_channels, dropout, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr='add', **kwargs)
        self.dropout = dropout
        self.W1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.W2 = nn.Linear(in_channels, out_channels, bias=bias)
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

        # TODO: maybe add edge weights to norm???

        # Start propagating messages
        out = self.propagate(edge_index, x=(x, x), norm=norm)

        # add self-message
        out += self.W1(x)

        # TODO: Is this message dropout??? --> I don't think it is...
        out = F.dropout(out, self.dropout, self.training)

        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm):    # TODO: incorporate the rating in this somehow (now it is assumed 1)
        """
        Implements message from node j to node i. To use extra args they must be passed in propagate().
        Using '_i' and '_j' variable names somehow associates that arg with the node.
        """
        return norm.view(-1, 1) * (self.W1(x_j) + self.W2(x_j * x_i))

    # Note: this is probably not needed because of aggr='add' in __init__()
    def aggregate(self, x, messages, index):
        out = torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
        return out


class NGCF(GNN_NCF):
    def __init__(self, node_feature_dim, gnn_hidden_layers=None,
                 item_emb=128, user_emb=128, mlp_dense_layers=None,
                 extra_emb_layers=False, dropout_rate=0.2):
        super(NGCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]  # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [128, 64, 64]  # default
        self.kwargs = {'node_feature_dim': node_feature_dim,
                       'gnn_hidden_layers': gnn_hidden_layers,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers}

        self.gnn_convs = nn.ModuleList(
            [NGCFConv(in_channels=node_feature_dim if i == 0 else gnn_hidden_layers[i - 1],
                      out_channels=gnn_hidden_layers[i],
                      dropout=dropout_rate)
             for i in range(len(gnn_hidden_layers))]
        )

        self.extra_emb_layers = extra_emb_layers

        if extra_emb_layers:
            self.item_embeddings = nn.Sequential(
                nn.Linear(sum(gnn_hidden_layers), item_emb),
                nn.ReLU()
            )
            self.user_embeddings = nn.Sequential(
                nn.Linear(sum(gnn_hidden_layers), user_emb),
                nn.ReLU()
            )

        self.MLP = build_MLP_layers(item_emb + user_emb if extra_emb_layers else sum(gnn_hidden_layers) * 2,
                                    mlp_dense_layers,
                                    dropout_rate=dropout_rate)

        # self.MLP = build_MLP_layers(gnn_hidden_layers[-1] * 2, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, GraphPointwiseDataset) or issubclass(dataset_class, GraphRankingDataset)

    def forward(self, *, graph, userIds, itemIds):
        # TODO: But in NCGF we are meant to embed items and users into the same space before applying the GNN layers!

        # encode all graph nodes with GNN
        combined_graph_emb = pass_gnn_layers(self.gnn_convs, graph, edge_weight=graph.edge_attr)

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
