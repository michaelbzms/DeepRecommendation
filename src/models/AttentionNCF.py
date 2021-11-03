import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionNCF(nn.Module):
    def __init__(self, item_dim, item_embeddings_size=64, dropout_rate=0.2,
                 dense1=64, dense2=32, att_dense=16):
        super(AttentionNCF, self).__init__()
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_embeddings_size),
            # nn.ReLU()
        )
        self.AttentionNet = nn.Sequential(
            nn.Linear(2*item_embeddings_size, att_dense),
            nn.ReLU(),
            nn.Linear(att_dense, 1)
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

    def forward(self, candidate_items, rated_items, user_matrix):
        I = rated_items.shape[0]      # == user_matrix.shape[1]
        B = candidate_items.shape[0]  # == user_matrix.shape[0]
        # item part
        candidate_embeddings = self.item_embeddings(candidate_items)   # (B, E)

        # attention between candidate items and rated items
        rated_embeddings = self.item_embeddings(rated_items).detach()  # (I, E)
        attention_scores = torch.zeros((B, I)).to(device)              # (B, I)
        for i in range(I):
            repeated_rated_embedding = rated_embeddings[i].view(1, -1).repeat(B, 1)   # repeat batch-size times for i-th rated item -> (B, E)
            item_emb_pairs = torch.cat((candidate_embeddings.detach(), repeated_rated_embedding), dim=1)
            attention_scores[:, [i]] = self.AttentionNet(item_emb_pairs)
        # pass through softmax
        attention_scores = F.softmax(attention_scores, dim=1)   # (B, I)

        # user part
        attended_user_matrix = torch.mul(attention_scores, user_matrix)
        user_embeddings = torch.matmul(attended_user_matrix, rated_embeddings)

        # combine
        combined = torch.cat((candidate_embeddings, user_embeddings), dim=1)
        # MLP part
        out = self.MLP(combined)
        return out
