import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax, subgraph
import numpy as np

from neural_collaborative_filtering.datasets.gnn_datasets import GraphPointwiseDataset, GraphRankingDataset
from neural_collaborative_filtering.models.base import GNN_NCF
from neural_collaborative_filtering.util import build_MLP_layers


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
            self.user2item_AttNet = nn.Sequential(
                nn.Linear(in_channels * 2, 1),
            )
            self.item2user_AttNet = nn.Sequential(
                nn.Linear(in_channels * 2, 1),
            )
            nn.init.xavier_uniform_(self.user2item_W[0].weight)
            nn.init.xavier_uniform_(self.item2user_W[0].weight)
        else:
            self.W = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout)
            )
            self.AttNet = nn.Sequential(
                nn.Linear(in_channels * 2, 1),
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
        # transform and attention input
        a_input = torch.cat([x_j, x_i], dim=1)
        if type == 'user2item':
            W = self.user2item_W
            a_scores = self.user2item_AttNet(a_input)
        elif type == 'item2user':
            W = self.item2user_W
            a_scores = self.item2user_AttNet(a_input)
        elif type == 'combined':
            W = self.W
            a_scores = self.AttNet(a_input)
        else:
            raise ValueError('Unrecognized message type in GNN')
        # calculate softmax
        a_scores = softmax(a_scores, index=to_index)
        # calculate all messages
        if weight is not None:
            messages = weight.view(-1, 1) * a_scores * W(x_j)
        else:
            messages = a_scores * W(x_j)
        return messages


class GraphNCF(GNN_NCF):
    def __init__(self, item_dim, user_dim, num_gnn_layers: int, hetero, node_emb=64, mlp_dense_layers=None,
                 dropout_rate=0.2, use_dot_product=False, concat=False,
                 message_dropout=None, node_dropout=None, convType='LightGCN'):
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
            'node_dropout': node_dropout,
            'hetero': hetero,
            'concat': concat,
            'convType': convType
        }
        self.concat = concat
        self.message_dropout = message_dropout
        self.node_dropout = node_dropout

        # embeddings layers
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, node_emb)
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(user_dim, node_emb)
        )

        # Light GCN convolutions to fine-tune previous embeddings using the graph
        # Note: Experimentally found that using the same weights on each layer works better
        self.convType = convType
        if convType == 'LightGCN':
            conv = LightGCNConv(in_channels=node_emb,
                                out_channels=node_emb,
                                dropout=dropout_rate / 2,
                                hetero=hetero)
        elif convType == 'LightGAT':
            conv = LightGATConv(in_channels=node_emb,
                                out_channels=node_emb,
                                dropout=dropout_rate / 2,
                                hetero=hetero)
        else:
            raise ValueError('Invalid convType.')
        self.gnn_convs = nn.ModuleList([conv for _ in range(num_gnn_layers)])   # same weights on every time step

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
        """
        Randomly remove p% of undirected edges (i.e. both directed edges between two nodes) in the graph.
        """
        if self.message_dropout is None or self.message_dropout <= 0.0:
            return user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr
        # message dropout -> randomly ignore p % of edges in the graph
        # Note: assumes edge attr non zero when using all edges (aka binary=False)
        if user2item_edge_index.shape[1] == item2user_edge_index.shape[1] and (user2item_edge_attr is not None and item2user_edge_attr is not None):
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

    def _node_dropout(self, num_nodes, except_node_ids, user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr):
        """
        Randomly remove p% of graph nodes (along with all their edges) that are not in the current batch themselves.
        Note: must be in k-hop neighbours to affect nodes in current batch. Can't control that.
        """
        if self.node_dropout is None or self.node_dropout <= 0.0:
            return user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr
        # subset MUST contain all node ids in the current batch (i.e. `except_node_ids`) because we need their embedding
        subset = np.random.choice(np.array([i for i in range(num_nodes) if i not in except_node_ids]),
                                  size=int((1.0 - self.node_dropout) * (num_nodes - len(except_node_ids))),
                                  replace=False).tolist()
        subset = torch.cat([torch.LongTensor(subset), torch.LongTensor(list(except_node_ids))])
        # get the edges for the induced subgraph that has (1-p) % of the nodes still in it
        user2item_edge_index, user2item_edge_attr = subgraph(subset, user2item_edge_index, user2item_edge_attr, num_nodes=num_nodes)
        item2user_edge_index, item2user_edge_attr = subgraph(subset, item2user_edge_index, item2user_edge_attr, num_nodes=num_nodes)
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

        # apply node dropout ONCE for ALL graph convolutions
        if self.node_dropout is not None and self.node_dropout > 0.0 and self.training:
            user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr = self._node_dropout(
                graph_emb.shape[0], set(torch.unique(torch.cat((itemIds, userIds))).tolist()),
                user2item_edge_index, item2user_edge_index, user2item_edge_attr, item2user_edge_attr
            )

        # apply message dropout ONCE for ALL graph convolutions
        if self.message_dropout is not None and self.message_dropout > 0.0 and self.training:
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
        _pos = pos_df.loc[input_pos]['pos'].values     # Note: could raise KeyError if an edge is missing!
        # create mask
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
        mask[_pos] = False
        # apply mask
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        return edge_index, edge_attr
