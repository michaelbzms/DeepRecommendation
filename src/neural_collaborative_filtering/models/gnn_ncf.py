import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax

from neural_collaborative_filtering.datasets.gnn_datasets import GraphPointwiseDataset, GraphRankingDataset
from neural_collaborative_filtering.models.base import GNN_NCF
from neural_collaborative_filtering.util import build_MLP_layers


# Note: Depricated
# class NGCFConv(MessagePassing):
#     """
#     NGCF Conv layer implementation taken and modified from:
#     https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377
#     """
#     def __init__(self, in_channels, out_channels,
#                  dropout=0.1, message_dropout=0.1,
#                  bias=True, **kwargs):
#         super(NGCFConv, self).__init__(aggr='add', **kwargs)
#         self.message_dropout = message_dropout
#         self.W1 = nn.Sequential(
#             nn.Linear(in_channels, out_channels, bias=bias),
#             nn.Dropout(dropout)
#         )
#         self.W2 = nn.Sequential(
#             nn.Linear(in_channels, out_channels, bias=bias),
#             nn.Dropout(dropout)
#         )
#         self.init_parameters()
#
#     def init_parameters(self):
#         nn.init.xavier_uniform_(self.W1[0].weight)
#         nn.init.xavier_uniform_(self.W2[0].weight)
#
#     def forward(self, x, edge_index, edge_attr=None):
#         if self.message_dropout is not None and self.training:
#             # message dropout -> randomly ignore p % of edges in the graph
#             mask = F.dropout(torch.ones(edge_index.shape[1]), self.message_dropout, self.training) > 0
#             edge_index_to_use = edge_index[:, mask]
#             edge_attr_to_use = edge_attr[mask] if edge_attr is not None else None
#             del mask
#         else:
#             edge_index_to_use = edge_index
#             edge_attr_to_use = edge_attr
#
#         # Compute normalization 1/sqrt{|Ni|*|Nj|} where i = from_ and j = to_
#         from_, to_ = edge_index_to_use
#         deg = degree(to_, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
#         del from_, to_, deg_inv_sqrt, deg
#
#         # Start propagating messages
#         out = self.propagate(edge_index_to_use, x=(x, x), norm=norm, weight=edge_attr_to_use)
#         if self.message_dropout is not None and self.training:
#             del edge_index_to_use, edge_attr_to_use
#         del norm
#
#         # add self-message
#         out += self.W1(x)
#
#         return F.leaky_relu(out)
#
#     def message(self, x_j, x_i, norm, weight):
#         """
#         Implements message from node j to node i. To use extra args they must be passed in propagate().
#         Using '_i' and '_j' variable names somehow associates that arg with the node.
#         Both norm and the rest are a vector of 1d length [num_edges].
#         """
#         # calculate all messages
#         if weight is not None:
#             messages = weight.view(-1, 1) * norm.view(-1, 1) * (self.W1(x_j) + self.W2(x_j * x_i))
#         else:
#             messages = norm.view(-1, 1) * (self.W1(x_j) + self.W2(x_j * x_i))
#         return messages


class LightGCNConv(MessagePassing):
    """
    LightGCN Conv layer implementation taken and modified from:
    https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377
    """
    def __init__(self, in_channels, out_channels, hetero, dropout=0.1, **kwargs):
        super(LightGCNConv, self).__init__(aggr='add', **kwargs)
        self.hetero = hetero
        if hetero:
            self.user2item_W = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout)
            )
            self.item2user_W = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout)
            )
            nn.init.xavier_uniform_(self.user2item_W[0].weight)
            nn.init.xavier_uniform_(self.item2user_W[0].weight)
        else:
            self.W = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout)
            )
            nn.init.xavier_uniform_(self.W[0].weight)

    def forward(self, x, user2item_edge_index, item2user_edge_index, user2item_edge_attr=None, item2user_edge_attr=None):
        # combine all edges into one
        total_edges = torch.cat([user2item_edge_index, item2user_edge_index], dim=1)
        total_edge_attr = None
        if user2item_edge_attr is not None and item2user_edge_attr is not None:
            total_edge_attr = torch.cat([user2item_edge_attr, item2user_edge_attr], dim=0)

        # Compute normalization 1/sqrt{|Ni|*|Nj|} where i = from_ and j = to_
        from_, to_ = total_edges
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if self.hetero:
            # propagate user -> item messages
            user2item_norm = deg_inv_sqrt[user2item_edge_index[0]] * deg_inv_sqrt[user2item_edge_index[1]]
            out1 = self.propagate(user2item_edge_index, x=x, norm=user2item_norm, weight=user2item_edge_attr, type='user2item')
            del user2item_norm
            # propagate item -> user messages
            item2user_norm = deg_inv_sqrt[item2user_edge_index[0]] * deg_inv_sqrt[item2user_edge_index[1]]
            out2 = self.propagate(item2user_edge_index, x=x, norm=item2user_norm, weight=item2user_edge_attr, type='item2user')
            del item2user_norm
            # add the two messages
            out = out1 + out2
            del from_, to_, deg_inv_sqrt, deg
        else:
            # calculate one norm
            norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
            del from_, to_, deg_inv_sqrt, deg
            # propagate messages
            out = self.propagate(total_edges, x=x, weight=total_edge_attr, norm=norm, type='combined')
            del norm

        return out

    def message(self, x_j, weight, norm, type: str):
        """
        Implements message from node j to node i. To use extra args they must be passed in propagate().
        Using '_i' and '_j' variable names somehow associates that arg with the node.
        Both norm and the rest are a vector of 1d length [num_edges].
        """
        # transform
        if type == 'user2item':
            W = self.user2item_W
        elif type == 'item2user':
            W = self.item2user_W
        elif type == 'combined':
            W = self.W
        else:
            raise ValueError('Unrecognized message type in GNN')
        # calculate all messages
        if weight is not None:
            messages = weight.view(-1, 1) * norm.view(-1, 1) * W(x_j)
        else:
            messages = norm.view(-1, 1) * W(x_j)
        return messages


class LightGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, hetero, dropout=0.1, **kwargs):
        super(LightGATConv, self).__init__(aggr='add', **kwargs)
        self.hetero = hetero
        if hetero:
            self.user2item_W = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout)
            )
            self.item2user_W = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout)
            )
            nn.init.xavier_uniform_(self.user2item_W[0].weight)
            nn.init.xavier_uniform_(self.item2user_W[0].weight)
        else:
            self.W = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout)
            )
            nn.init.xavier_uniform_(self.W[0].weight)

    def forward(self, x, user2item_edge_index, item2user_edge_index, user2item_edge_attr=None,
                item2user_edge_attr=None):
        if self.hetero:
            # propagate user -> item messages
            out1 = self.propagate(user2item_edge_index, x=(x, x), weight=user2item_edge_attr,
                                  to_index=user2item_edge_index[1], type='user2item')
            # propagate item -> user messages
            out2 = self.propagate(item2user_edge_index, x=(x, x), weight=item2user_edge_attr,
                                  to_index=item2user_edge_index[1], type='item2user')
            # add the two messages
            out = out1 + out2
        else:
            # combine all edges into one
            total_edges = torch.cat([user2item_edge_index, item2user_edge_index], dim=1)
            total_edge_attr = None
            if user2item_edge_attr is not None and item2user_edge_attr is not None:
                total_edge_attr = torch.cat([user2item_edge_attr, item2user_edge_attr], dim=0)
            # propagate messages
            out = self.propagate(total_edges, x=(x, x), weight=total_edge_attr,
                                 to_index=total_edges[1], type='combined')

        return out

    def message(self, x_j, x_i, weight, to_index, type: str):
        """
        Implements message from node j to node i. To use extra args they must be passed in propagate().
        Using '_i' and '_j' variable names somehow associates that arg with the node.
        Weight is optionally a vector of 1d length [num_edges].
        """
        # transform
        if type == 'user2item':
            W = self.user2item_W
        elif type == 'item2user':
            W = self.item2user_W
        elif type == 'combined':
            W = self.W
        else:
            raise ValueError('Unrecognized message type in GNN')
        # calculate att weights
        a_scores = torch.bmm(x_j.unsqueeze(1), x_i.unsqueeze(2)).view(-1, 1)
        a_scores = softmax(a_scores, index=to_index)
        # calculate all messages
        if weight is not None:
            messages = weight.view(-1, 1) * a_scores * W(x_j)
        else:
            messages = a_scores * W(x_j)
        return messages


class GraphNCF(GNN_NCF):
    def __init__(self, item_dim, user_dim, num_gnn_layers: int, hetero, node_emb=64, mlp_dense_layers=None,
                 dropout_rate=0.2, use_dot_product=False, message_dropout=None, concat=False, convType='LightGAT'):
        super(GraphNCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]  # default
        self.kwargs = {
            'item_dim': item_dim,
            'user_dim': user_dim,
            'node_emb': node_emb,
            'num_gnn_layers': num_gnn_layers,
            'mlp_dense_layers': mlp_dense_layers,
            'use_dot_product': use_dot_product,     # overrides mlp_dense_layers
            'dropout_rate': dropout_rate,
            'message_dropout': message_dropout,
            'hetero': hetero,
            'concat': concat,
            'convType': convType
        }
        self.concat = concat
        self.message_dropout = message_dropout

        # Item embeddings layer
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, node_emb)
        )

        # User embeddings layer
        self.user_embeddings = nn.Sequential(
            nn.Linear(user_dim, node_emb)
        )

        # Light GCN convolutions to fine-tune previous embeddings using the graph
        self.convType = convType
        if convType == 'LightGCN':
            self.gnn_convs = nn.ModuleList(
                [LightGCNConv(in_channels=node_emb,
                              out_channels=node_emb,
                              dropout=dropout_rate / 2,
                              hetero=hetero)
                 for _ in range(num_gnn_layers)]
            )
        elif convType == 'LightGAT':
            self.gnn_convs = nn.ModuleList(
                [LightGATConv(in_channels=node_emb,
                              out_channels=node_emb,
                              dropout=dropout_rate / 2,
                              hetero=hetero)
                 for _ in range(num_gnn_layers)]
            )
        else:
            raise ValueError('Invalid convType.')

        # MLP layers or simply use dot product
        if use_dot_product:
            self.MLP = None
        else:
            self.MLP = build_MLP_layers(node_emb * (num_gnn_layers + 1) * 2 if self.concat else node_emb * 2,
                                        mlp_dense_layers,
                                        dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def important_hypeparams(self) -> str:
        return '_' + self.convType

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, GraphPointwiseDataset) or issubclass(dataset_class, GraphRankingDataset)

    def _message_dropout(self, user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr):
        if self.message_dropout is None: return
        # message dropout -> randomly ignore p % of edges in the graph
        if user2item_edge_index.shape[1] == item2user_edge_index.shape[1] and (
                user2item_edge_attr is not None and item2user_edge_attr is not None):  # TODO: assumes edge attr non zero when using all edges (aka binary=False)
            # assumes all edges are symmetrical (!)
            mask = F.dropout(torch.ones(user2item_edge_index.shape[1]), self.message_dropout, self.training) > 0
            # mask both user -> item and item -> user edge
            user2item_edge_index = user2item_edge_index[:, mask]
            item2user_edge_index = item2user_edge_index[:, mask]
            if user2item_edge_attr is not None:
                user2item_edge_attr = user2item_edge_attr[mask]
            if item2user_edge_attr is not None:
                item2user_edge_attr = item2user_edge_attr[mask]
            del mask
        else:
            # mask user -> item edges
            mask1 = F.dropout(torch.ones(user2item_edge_index.shape[1]), self.message_dropout, self.training) > 0
            user2item_edge_index = user2item_edge_index[:, mask1]
            if user2item_edge_attr is not None:
                user2item_edge_attr = user2item_edge_attr[mask1]
            del mask1
            # mask item -> user edges
            mask2 = F.dropout(torch.ones(item2user_edge_index.shape[1]), self.message_dropout, self.training) > 0
            item2user_edge_index = item2user_edge_index[:, mask2]
            if item2user_edge_attr is not None:
                item2user_edge_attr = item2user_edge_attr[mask2]
            del mask2

        return user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr

    def forward(self, graph, userIds, itemIds, device, mask_targets=True):
        # embed item and user input features
        item_emb = self.item_embeddings(graph.item_features)
        user_emb = self.user_embeddings(graph.user_features)

        # stack nodes with items first
        graph_emb = torch.vstack([item_emb, user_emb])
        del item_emb, user_emb   # remove any unnecessary memory

        # edge index and (optionally) attr to use
        user2item_edge_index = graph.user2item_edge_index
        item2user_edge_index = graph.item2user_edge_index
        user2item_edge_attr = graph.user2item_edge_attr if hasattr(graph, 'user2item_edge_attr') else None
        item2user_edge_attr = graph.item2user_edge_attr if hasattr(graph, 'item2user_edge_attr') else None

        # if training mask out edges that are currently targets in the batch
        if self.training and mask_targets:
            # mask user -> item edges
            _inp1 = torch.stack([userIds.cpu(), itemIds.cpu()]).T.tolist()
            user2item_edge_index, user2item_edge_attr = self._mask_edge_index(_inp1, graph.pos_df, user2item_edge_index, user2item_edge_attr, device)
            # mask item -> user edges
            _inp2 = torch.stack([itemIds.cpu(), userIds.cpu()]).T.tolist()
            item2user_edge_index, item2user_edge_attr = self._mask_edge_index(_inp1,  graph.pos_df, item2user_edge_index, item2user_edge_attr, device)

        # apply message dropout ONCE for ALL graph convolutions
        if self.message_dropout is not None and self.training:
            user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr = self._message_dropout(
                user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr
            )

        # encode all graph nodes with GNN
        hs = [graph_emb]
        for gnn_conv in self.gnn_convs:
            graph_emb = gnn_conv(
                graph_emb,
                user2item_edge_index=user2item_edge_index,
                item2user_edge_index=item2user_edge_index,
                user2item_edge_attr=user2item_edge_attr,
                item2user_edge_attr=item2user_edge_attr
            )
            hs.append(graph_emb)

        # aggregate all embeddings (including original one) into one node embedding
        if self.concat:
            combined_graph_emb = torch.cat(hs, dim=1)
        else:
            combined_graph_emb = torch.mean(torch.stack(hs, dim=0), dim=0)

        # find embeddings of items in batch
        item_emb = combined_graph_emb[itemIds]

        # find embeddings of users in batch
        user_emb = combined_graph_emb[userIds]

        if self.MLP is not None:
            # use these to forward the NCF model
            combined = torch.cat((item_emb, user_emb), dim=1)
            out = self.MLP(combined)
        else:
            # simple dot product
            out = torch.bmm(user_emb.unsqueeze(1), item_emb.unsqueeze(2)).view(-1, 1)

        return out

    def _mask_edge_index(self, input_pos, pos_df, edge_index, edge_attr, device):
        _pos = pos_df.loc[input_pos]['pos'].values     # TODO: could raise KeyError if an edge is missing
        # create mask
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
        mask[_pos] = False
        # apply mask
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        return edge_index, edge_attr


# Note: Depricated
# class NGCF(GNN_NCF):
#     def __init__(self, item_dim, user_dim, gnn_hidden_layers=None,
#                  node_emb=64, mlp_dense_layers=None, extra_emb_layers=True,
#                  dropout_rate=0.2, message_dropout=0.1):
#         super(NGCF, self).__init__()
#         if mlp_dense_layers is None: mlp_dense_layers = [256, 128]  # default
#         if gnn_hidden_layers is None: gnn_hidden_layers = [64, 64]  # default
#         self.kwargs = {'item_dim': item_dim,
#                        'user_dim': user_dim,
#                        'gnn_hidden_layers': gnn_hidden_layers,
#                        'node_emb': node_emb,
#                        'mlp_dense_layers': mlp_dense_layers,
#                        'extra_emb_layers': extra_emb_layers,
#                        'dropout_rate': dropout_rate,
#                        'message_dropout': message_dropout}
#
#         # optionally embed the user and item (fixed) input vectors before passing through GNNs
#         self.extra_emb_layers = extra_emb_layers
#         if extra_emb_layers:
#             self.item_embeddings = nn.Sequential(
#                 nn.Linear(item_dim, node_emb)
#             )
#             self.user_embeddings = nn.Sequential(
#                 nn.Linear(user_dim, node_emb)
#             )
#             gnn_input_dim = node_emb
#         else:
#             assert item_dim == user_dim, 'Error: Cannot use different sized embeddings for nodes of graph'
#             gnn_input_dim = item_dim   # == user_dim
#
#         # define GNN convolutions to apply
#         self.gnn_convs = nn.ModuleList(
#             [NGCFConv(in_channels=gnn_input_dim if i == 0 else gnn_hidden_layers[i - 1],
#                       out_channels=gnn_hidden_layers[i],
#                       dropout=dropout_rate,
#                       message_dropout=message_dropout)
#              for i in range(len(gnn_hidden_layers))]
#         )
#
#         # the NCF MLP network
#         # TODO: could also try just a simple dot product
#         self.MLP = build_MLP_layers(sum(gnn_hidden_layers) * 2,
#                                     mlp_dense_layers,
#                                     dropout_rate=dropout_rate)
#
#     def get_model_parameters(self) -> dict[str]:
#         return self.kwargs
#
#     def is_dataset_compatible(self, dataset_class):
#         return issubclass(dataset_class, GraphPointwiseDataset) or issubclass(dataset_class, GraphRankingDataset)
#
#     def forward(self, graph, userIds, itemIds, device):
#         if self.extra_emb_layers:
#             # embed item and user input features
#             item_emb = self.item_embeddings(graph.item_features)
#             user_emb = self.user_embeddings(graph.user_features)
#         else:
#             # or use the fixed input directly (shapes must match in this case)
#             assert graph.item_features.shape[1] == graph.user_features.shape[1], 'Error: Cannot use different sized embeddings for nodes of graph'
#             item_emb = graph.item_features
#             user_emb = graph.user_features
#
#         # stack nodes with items first
#         graph_emb = torch.vstack([item_emb, user_emb])
#         # remove any unnecessary memory
#         if self.extra_emb_layers:
#             del item_emb, user_emb
#         # encode all graph nodes with GNN
#         hs = []
#         for gnn_conv in self.gnn_convs:
#             graph_emb = gnn_conv(graph_emb, graph.edge_index, graph.edge_attr)
#             hs.append(graph_emb)
#
#         # concat all intermediate representations
#         combined_graph_emb = torch.cat(hs, dim=1)
#
#         # find embeddings of items in batch
#         item_emb = combined_graph_emb[itemIds.long()]
#
#         # find embeddings of users in batch
#         user_emb = combined_graph_emb[userIds.long()]
#
#         # use these to forward the NCF model
#         combined = torch.cat((item_emb, user_emb), dim=1)
#         out = self.MLP(combined)
#
#         return out
