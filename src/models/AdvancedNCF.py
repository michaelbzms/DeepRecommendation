import torch
from torch import nn

from models.NCF import NCF
from util import build_MLP_layers


class AdvancedNCF(NCF):
    def __init__(self, item_dim, item_emb=128, user_emb=128, mlp_dense_layers=None, dropout_rate=0.2):
        super(AdvancedNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [256, 128]         # default
        self.kwargs = {'item_dim': item_dim, 'item_emb': item_emb, 'user_emb': user_emb, 'mlp_dense_layers': mlp_dense_layers}
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb),
            nn.ReLU()
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(item_emb, user_emb),         # TODO: this inputs item_emb
            nn.ReLU()
        )
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def forward(self, candidate_items, item_matrix, user_matrix):
        # item part
        candidate_embeddings = self.item_embeddings(candidate_items)
        # user part
        rated_embeddings = self.item_embeddings(item_matrix)
        ratings_per_user = user_matrix.count_nonzero(dim=1)
        ratings_per_user[ratings_per_user == 0] = 1.0       # avoid div by zero
        user_input = torch.div(torch.matmul(user_matrix, rated_embeddings), ratings_per_user.view(-1, 1))
        # user part
        user_embeddings = self.user_embeddings(user_input)
        # combine
        combined = torch.cat((candidate_embeddings, user_embeddings), dim=1)
        # MLP part
        out = self.MLP(combined)
        return out
