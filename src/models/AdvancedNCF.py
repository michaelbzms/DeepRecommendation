import torch
from torch import nn


class AdvancedNCF(nn.Module):
    def __init__(self, item_dim, item_embeddings_size=128, dropout_rate=0.2,
                 dense1=128, dense2=64):
        super(AdvancedNCF, self).__init__()
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_embeddings_size),
            nn.ReLU()
        )
        self.MLP = nn.Sequential(
            nn.Linear(2*item_embeddings_size, dense1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense2, 1)
        )

    def forward(self, candidate_items, item_matrix, user_matrix):
        # item part
        candidate_embeddings = self.item_embeddings(candidate_items)
        # user part
        rated_embeddings = self.item_embeddings(item_matrix).detach()   # (!) detach
        ratings_per_user = user_matrix.count_nonzero(dim=1)
        ratings_per_user[ratings_per_user == 0] = 1.0       # avoid div by zero
        user_embeddings = torch.div(torch.matmul(user_matrix, rated_embeddings), ratings_per_user.view(-1, 1))
        # combine
        combined = torch.cat((candidate_embeddings, user_embeddings), dim=1)
        # MLP part
        out = self.MLP(combined)
        return out