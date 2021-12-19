import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn.functional as F

from gnns.datasets.GNN_dataset import GNN_Dataset
from gnns.models.GNN import GNN_NCF
from neural_collaborative_filtering.util import build_MLP_layers


class GAT_NCF(GNN_NCF):
    def __init__(self, gnn_hidden_layers=None, item_emb=128, user_emb=128, mlp_dense_layers=None, dropout_rate=0.2):
        super(GAT_NCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]    # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [128]       # default
        self.kwargs = {'gnn_hidden_layers': gnn_hidden_layers,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers}

        self.gnn_convs = nn.ModuleList(
            [GATConv(in_channels=-1 if i == 0 else gnn_hidden_layers[i-1],
                     out_channels=gnn_hidden_layers[i],
                     edge_dim=1) for i in range(len(gnn_hidden_layers))]
        )

        self.item_embeddings = nn.Sequential(
            nn.Linear(sum(gnn_hidden_layers), item_emb),
            nn.ReLU()
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(sum(gnn_hidden_layers), user_emb),
            nn.ReLU()
        )
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

        # self.MLP = build_MLP_layers(gnn_hidden_layers[-1] * 2, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, GNN_Dataset)

    def forward(self, graph, userIds, itemIds):  # needs to be True for training only I think
        # encode with GNN
        graph_emb = graph.x        # initial representation
        hs = []
        for gnn_conv in self.gnn_convs:
            graph_emb = gnn_conv(graph_emb, graph.edge_index, edge_attr=graph.edge_attr)
            graph_emb = F.leaky_relu(graph_emb)
            hs.append(graph_emb)
        combined_graph_emb = torch.cat(hs, dim=1)
        # TODO: keep only the latest or keep previous (e.g. concat or use LSTM/GRU)

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

