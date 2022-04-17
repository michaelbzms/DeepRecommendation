import torch
import torch_scatter
import random
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from neural_collaborative_filtering.datasets.gnn_datasets import GraphPointwiseDataset, GraphRankingDataset
from neural_collaborative_filtering.models.base import GNN_NCF
from neural_collaborative_filtering.util import build_MLP_layers


class NGCFConv(MessagePassing):
    """
    NGCF Conv layer implementation taken and modified from:
    https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377
    """
    def __init__(self, in_channels, out_channels,
                 dropout=0.1, message_dropout=None,
                 bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr='add', **kwargs)
        self.message_dropout = message_dropout
        self.W1 = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.Dropout(dropout)
        )
        self.W2 = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.Dropout(dropout)
        )
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W1[0].weight)
        nn.init.xavier_uniform_(self.W2[0].weight)

    def forward(self, x, edge_index, edge_attr=None):
        if self.message_dropout is not None:
            # TODO: this is way too slow (4-5 times slower than without it
            # message dropout -> randomly ignore p % of edges in the graph i.e. keep only (1-p) % of them
            random_keep_inx = random.sample(range(edge_index.shape[1]), int((1.0 - self.message_dropout) * edge_index.shape[1]))
            edge_index_to_use = edge_index[:, random_keep_inx]
            edge_attr_to_use = edge_attr[random_keep_inx] if edge_attr is not None else None
        else:
            edge_index_to_use = edge_index
            edge_attr_to_use = edge_attr

        # Compute normalization 1/sqrt{|Ni|*|Nj|} where i = from_ and j = to_
        from_, to_ = edge_index_to_use
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages
        out = self.propagate(edge_index_to_use, x=(x, x), norm=norm, weight=edge_attr_to_use)

        # add self-message
        out += self.W1(x)

        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm, weight):    # TODO: incorporate the rating in this somehow (now it is assumed 1)
        """
        Implements message from node j to node i. To use extra args they must be passed in propagate().
        Using '_i' and '_j' variable names somehow associates that arg with the node.
        Both norm and the rest are a vector of 1d length [num_edges].
        """
        # calculate all messages
        if weight is not None:
            messages = weight.view(-1, 1) * norm.view(-1, 1) * (self.W1(x_j) + self.W2(x_j * x_i))
        else:
            messages = norm.view(-1, 1) * (self.W1(x_j) + self.W2(x_j * x_i))
        return messages


class NGCF(GNN_NCF):
    def __init__(self, item_dim, user_dim, gnn_hidden_layers=None,
                 node_emb=64, mlp_dense_layers=None, extra_emb_layers=True,
                 dropout_rate=0.2, gnn_dropout_rate=0.1):
        super(NGCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]  # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [64, 64]  # default
        self.kwargs = {'item_dim': item_dim,
                       'user_dim': user_dim,
                       'gnn_hidden_layers': gnn_hidden_layers,
                       'node_emb': node_emb,
                       'mlp_dense_layers': mlp_dense_layers}

        # optionally embed the user and item (fixed) input vectors before passing through GNNs
        self.extra_emb_layers = extra_emb_layers
        if extra_emb_layers:
            self.item_embeddings = nn.Sequential(
                nn.Linear(item_dim, node_emb),
                nn.ReLU()
            )
            self.user_embeddings = nn.Sequential(
                nn.Linear(user_dim, node_emb),
                nn.ReLU()
            )
            gnn_input_dim = node_emb
        else:
            assert item_dim == user_dim, 'Error: Cannot use different sized embeddings for nodes of graph'
            gnn_input_dim = item_dim   # == user_dim

        # define GNN convolutions to apply
        self.gnn_convs = nn.ModuleList(
            [NGCFConv(in_channels=gnn_input_dim if i == 0 else gnn_hidden_layers[i - 1],
                      out_channels=gnn_hidden_layers[i],
                      dropout=gnn_dropout_rate)
             for i in range(len(gnn_hidden_layers))]
        )

        # the NCF MLP network
        # TODO: could also try just a simple dot product
        self.MLP = build_MLP_layers(sum(gnn_hidden_layers) * 2,
                                    mlp_dense_layers,
                                    dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, GraphPointwiseDataset) or issubclass(dataset_class, GraphRankingDataset)

    def forward(self, graph, userIds, itemIds):
        if self.extra_emb_layers:
            # embed item and user input features
            item_emb = self.item_embeddings(graph.item_features)
            user_emb = self.user_embeddings(graph.user_features)
        else:
            # or use the fixed input directly (shapes must match in this case)
            assert graph.item_features.shape[1] == graph.user_features.shape[1], 'Error: Cannot use different sized embeddings for nodes of graph'
            item_emb = graph.item_features
            user_emb = graph.user_features

        # encode all graph nodes with GNN
        graph_emb = torch.vstack([item_emb, user_emb])          # stack nodes with items first!
        hs = []
        for gnn_conv in self.gnn_convs:
            graph_emb = gnn_conv(graph_emb, graph.edge_index, graph.edge_attr)   # TODO: add weights
            graph_emb = F.leaky_relu(graph_emb)
            hs.append(graph_emb)

        # concat all intermediate representations
        combined_graph_emb = torch.cat(hs, dim=1)

        # find embeddings of items in batch
        item_emb = combined_graph_emb[itemIds.long()]

        # find embeddings of users in batch
        user_emb = combined_graph_emb[userIds.long()]

        # use these to forward the NCF model
        combined = torch.cat((item_emb, user_emb), dim=1)
        out = self.MLP(combined)

        return out
