import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
import torch.nn.functional as F

from neural_collaborative_filtering.datasets.gnn_datasets import GNN_Dataset
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


class GCN_NCF(GNN_NCF):
    def __init__(self, gnn_hidden_layers=None, item_emb=128, user_emb=128, mlp_dense_layers=None, dropout_rate=0.2):
        super(GCN_NCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]    # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [128, 64, 64]       # default
        self.kwargs = {'gnn_hidden_layers': gnn_hidden_layers,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers}

        self.gnn_convs = nn.ModuleList(
            [GCNConv(in_channels=-1 if i == 0 else gnn_hidden_layers[i-1],
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
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]         # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [128]            # default
        self.kwargs = {'gnn_hidden_layers': gnn_hidden_layers,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers,
                       'num_heads': num_heads,
                       'extra_emb_layers': extra_emb_layers,
                       'initial_repr_dim': initial_repr_dim,
                       'edge_dim': edge_dim}

        self.gnn_convs = nn.ModuleList(
            [GATConv(in_channels=initial_repr_dim if i == 0 else gnn_hidden_layers[i-1],
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
            self.MLP = build_MLP_layers(sum(gnn_hidden_layers) * num_heads * 2, mlp_dense_layers, dropout_rate=dropout_rate)

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


class GAT_NCF_Multimodal(GNN_NCF):
    def __init__(self, initial_repr_dim=-1, gnn_hidden_layers=None,
                 mlp_dense_layers=None, att_dense=128, num_heads=1,
                 dropout_rate=0.2, edge_dim=-1):
        super(GAT_NCF_Multimodal, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]    # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [128]       # default
        self.kwargs = {'gnn_hidden_layers': gnn_hidden_layers,
                       'mlp_dense_layers': mlp_dense_layers,
                       'num_heads': num_heads,
                       'initial_repr_dim': initial_repr_dim,
                       'edge_dim': edge_dim,
                       'att_dense': att_dense}

        self.gnn_convs = nn.ModuleList(
            [GATConv(in_channels=initial_repr_dim if i == 0 else gnn_hidden_layers[i-1],
                     out_channels=gnn_hidden_layers[i],
                     edge_dim=edge_dim,
                     heads=num_heads) for i in range(len(gnn_hidden_layers))]
        )

        node_emb_dim = sum(gnn_hidden_layers) * num_heads  # TODO: num_heads???

        self.W_query1 = nn.Linear(node_emb_dim, att_dense)
        self.W_key1 = nn.Linear(node_emb_dim, att_dense)
        self.W_value1 = nn.Linear(node_emb_dim, att_dense)
        self.MultiHeadAttLayer = nn.MultiheadAttention(1, num_heads=num_heads, batch_first=True)  # TODO: dropout?

        self.W_query2 = nn.Linear(node_emb_dim, att_dense)
        self.W_key2 = nn.Linear(node_emb_dim, att_dense)
        self.W_value2 = nn.Linear(node_emb_dim, att_dense)
        self.MultiHeadAttLayer2 = nn.MultiheadAttention(1, num_heads=num_heads, batch_first=True)  # TODO: dropout?

        self.MLP = build_MLP_layers(sum(gnn_hidden_layers) * num_heads * 2, mlp_dense_layers, dropout_rate=dropout_rate)

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

        # item --> user attention
        Q1 = torch.unsqueeze(self.W_query1(item_emb), 2)
        K1 = torch.unsqueeze(self.W_key1(user_emb), 2)
        V1 = torch.unsqueeze(self.W_value1(user_emb), 2)
        att_emb1, att_weights1 = self.MultiHeadAttLayer(Q1, K1, V1)
        att_emb1 = torch.squeeze(att_emb1, 2)

        # user --> item attention
        Q2 = torch.unsqueeze(self.W_query2(user_emb), 2)
        K2 = torch.unsqueeze(self.W_key2(item_emb), 2)
        V2 = torch.unsqueeze(self.W_value2(item_emb), 2)
        att_emb2, att_weights2 = self.MultiHeadAttLayer(Q2, K2, V2)
        att_emb2 = torch.squeeze(att_emb2, 2)

        combined = torch.cat((att_emb1, att_emb2), dim=1)

        out = self.MLP(combined)

        return out
