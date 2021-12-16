import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn.functional as F

from gnns.models.GNN_NCF import GNN_NCF
from neural_collaborative_filtering.util import build_MLP_layers


class GCN_NCF(GNN_NCF):
    def __init__(self, gnn_hidden_layers=None, item_emb=128, user_emb=128, mlp_dense_layers=None, dropout_rate=0.2):
        super(GCN_NCF, self).__init__()
        if mlp_dense_layers is None: mlp_dense_layers = [256, 128]         # default
        if gnn_hidden_layers is None: gnn_hidden_layers = [256, 128]       # default
        self.kwargs = {'gnn_hidden_layers': gnn_hidden_layers,
                       'item_emb': item_emb,
                       'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers}

        self.gnn_convs = nn.ModuleList(
            [GCNConv(in_channels=-1 if i == 0 else gnn_hidden_layers[i-1], out_channels=gnn_hidden_layers[i]) for i in range(len(gnn_hidden_layers))]
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

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def forward(self, graph, batch, remove_edges_if_target: bool = False):  # needs to be True for training only I think
        if remove_edges_if_target:
            # Temporarily remove edges in graph that we are about to predict  -> in graph.edge_index
            # TODO: can I speed this up?
            black_list = set([(int(x), int(y)) for x, y in zip(batch[0], batch[1])]) \
                .union(set([(int(y), int(x)) for x, y in zip(batch[0], batch[1])]))
            edge_index = [[], []]
            edge_attr = []
            masked_out = 0
            for i in range(len(graph.edge_index[0])):
                if (int(graph.edge_index[0][i]), int(graph.edge_index[1][i])) not in black_list:
                    edge_index[0].append(int(graph.edge_index[0][i]))
                    edge_index[1].append(int(graph.edge_index[1][i]))
                    edge_attr.append(graph.edge_attr[i])
                else:
                    masked_out += 1
            assert(masked_out == 2 * len(batch[2]))  # assert everything is going as planned
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

        # encode with GNN
        graph_emb = graph.x        # initial representation
        for gnn_conv in self.gnn_convs:
            graph_emb = gnn_conv(graph_emb, edge_index, edge_weight=edge_attr)
            graph_emb = F.leaky_relu(graph_emb)
        print(graph_emb)
        # TODO: keep only the latest or keep previous (e.g. concat or use LSTM/GRU)

        # find embeddings of items in batch
        item_emb = graph_emb[batch[1]]

        # find embeddings of users in batch
        user_emb = graph_emb[graph.num_items + batch[0]]

        # use these to forward the NCF model
        item_emb = self.item_embeddings(item_emb)
        user_emb = self.user_embeddings(user_emb)
        combined = torch.cat((item_emb, user_emb), dim=1)
        out = self.MLP(combined)

        return out

