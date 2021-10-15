import torch
from torch import nn


class BasicNCF(nn.Module):
    def __init__(self, item_dim, user_dim, dropout_rate=0.2,
                 item_embeddings_size=512, user_embeddings_size=512,
                 dense1=512, dense2=128):
        super(BasicNCF, self).__init__()
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_embeddings_size),
            nn.ReLU()
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(user_dim, user_embeddings_size),
            nn.ReLU()
        )
        self.MLP = nn.Sequential(
            nn.Linear(item_embeddings_size + user_embeddings_size, dense1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense2, 1)
        )

    def forward(self, X_item, X_user):
        item_emb = self.item_embeddings(X_item)
        user_emb = self.user_embeddings(X_user)
        combined = torch.cat((item_emb, user_emb), dim=1)
        out = self.MLP(combined)
        return out
