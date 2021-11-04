import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionNCF(nn.Module):
    def __init__(self, item_dim, dropout_rate=0.2,
                 dense1=32, dense2=20, dense3=12, att_dense1=8):
        super(AttentionNCF, self).__init__()
        self.AttentionNet = nn.Sequential(
            nn.Linear(2*item_dim, 1),
            # nn.ReLU(),
            # nn.Linear(att_dense1, 1)
        )
        self.MLP = nn.Sequential(
            nn.Linear(2*item_dim, dense1),
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
        # item part
        # candidate_embeddings = self.item_embeddings(candidate_items)   # (B, E)
        # attention between candidate items and rated items
        # rated_embeddings = self.item_embeddings(rated_items).detach()  # (I, E)
        attention_scores = torch.zeros((B, I)).to(device)              # (B, I)
        # for i in range(I):
        #     repeated_rated_items = rated_items[i].view(1, -1).repeat(B, 1)   # repeat batch-size times for i-th rated item -> (B, E)
        #     item_pairs = torch.cat((candidate_items, repeated_rated_items), dim=1)
        #     attention_scores[:, [i]] = self.AttentionNet(item_pairs)
        # # TODO: is this equivalent but with a smaller for-loop?
        # for i in range(B):
        #     repeated_candidate_item = candidate_items[i].view(1, -1).repeat(I, 1)   # (I, F)
        #     item_pairs = torch.cat((repeated_candidate_item, rated_items), dim=1)   # (I, 2*F)
        #     attention_scores[[i], :] = self.AttentionNet(item_pairs).view(1, -1)    # (I, 1)  -> (1, I)

        # TODO: the one that interleaves matters! I think this works correctly into (B, I) shape because the first I elements contain all different rated items and they become the first row of length I
        attNetInput = torch.cat((candidate_items.repeat_interleave(I, dim=0), rated_items.repeat(B, 1)), dim=1)
        attention_scores = self.AttentionNet(attNetInput).view(B, I)
        # pass through softmax
        attention_scores = F.softmax(attention_scores, dim=1)   # (B, I)

        # user part
        attended_user_matrix = torch.mul(attention_scores, user_matrix)
        user_embeddings = torch.matmul(attended_user_matrix, rated_items)

        # combine
        combined = torch.cat((candidate_items, user_embeddings), dim=1)
        # MLP part
        out = self.MLP(combined)
        return out
