import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionNCF(nn.Module):
    def __init__(self, item_dim, dropout_rate=0.2,
                 item_emb=512, user_emb=1024,
                 dense1=1024, dense2=512, dense3=256):
        super(AttentionNCF, self).__init__()
        self.ItemEmbeddings = nn.Sequential(
            nn.Linear(item_dim, 2*item_emb),
            nn.ReLU(),
            nn.Linear(2*item_emb, item_emb),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        self.UserEmbeddings = nn.Sequential(
            nn.Linear(item_dim, 2*user_emb),
            nn.ReLU(),
            nn.Linear(2*user_emb, user_emb),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        self.AttentionNet = nn.Sequential(
            nn.Linear(2*item_dim, 1)
        )
        self.MLP = nn.Sequential(
            nn.Linear(item_emb + user_emb, dense1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense2, dense3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense3, 1)
        )

    def forward(self, candidate_items, rated_items, user_matrix):
        I = rated_items.shape[0]      # == user_matrix.shape[1]
        B = candidate_items.shape[0]  # == user_matrix.shape[0]

        # attention on rated items
        """ Note: the one that interleaves matters! I think this works correctly into (B, I) shape 
        because the first I elements contain all different rated items and they become the first row of length I """
        attNetInput = torch.cat((candidate_items.repeat_interleave(I, dim=0), rated_items.repeat(B, 1)), dim=1)
        attention_scores = self.AttentionNet(attNetInput).view(B, I)
        # mask unrated items per user (!) - otherwise there may be high weights on 0 entries
        attention_scores[user_matrix == 0.0] = -float('inf')    # so that softmax gives this a 0 attention weight
        # pass through softmax
        attention_scores = F.softmax(attention_scores, dim=1)   # (B, I)
        attention_scores = attention_scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # will get NaNs if a user has 0 ratings. Replace those with 0

        # aggregate item features based on ratings and attention weights
        attended_user_matrix = torch.mul(attention_scores, user_matrix)
        user_estimated_features = torch.matmul(attended_user_matrix, rated_items)

        # pass through item embeddings layer
        item_embeddings = self.ItemEmbeddings(candidate_items)
        # pass through user embeddings layer
        user_embeddings = self.UserEmbeddings(user_estimated_features)

        # combine
        combined = torch.cat((item_embeddings, user_embeddings), dim=1)

        # MLP part
        out = self.MLP(combined)

        return out
