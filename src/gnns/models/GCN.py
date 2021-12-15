import torch
from torch import nn
from torch.fx import GraphModule
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero, HeteroConv, GATConv

from gnns.models.GNN_NCF import GNN_NCF
from neural_collaborative_filtering.util import build_MLP_layers


class GCN_NCF(GNN_NCF):
    def __init__(self, gnn_hidden=256, item_emb=128, user_emb=128, mlp_dense_layers=None, dropout_rate=0.2):
        super(GCN_NCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]         # default
        self.kwargs = {'gnn_hidden': gnn_hidden,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers}

        self.conv = HeteroConv({
            ('user', 'rates', 'item'): GATConv(in_channels=(-1, -1), out_channels=gnn_hidden, edge_dim=1),
            ('item', 'ratedby', 'user'): GATConv(in_channels=(-1, -1), out_channels=gnn_hidden, edge_dim=1)
        })

        self.item_embeddings = nn.Sequential(
            nn.Linear(0, item_emb),
            nn.ReLU()
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(0, user_emb),
            nn.ReLU()
        )
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def forward(self, graph, batch):
        # encode with GNN
        graph_emb = self.conv(graph.x_dict, graph.edge_index_dict,  **{'edge_attr_dict': graph.edge_attr_dict})  # this works for adding weigths

        print(graph_emb)
        print(graph_emb)
        # use these embeddings forward
        pass