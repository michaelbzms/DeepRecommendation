import torch
from torch import nn
import torch.nn.functional as F

from models.NCF import NCF
from plots import visualize_attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionNCF(NCF):
    def __init__(self, item_dim, item_emb=256, user_emb=256, att_dense=None,
                 dense1=1024, dense2=512, dense3=256, dense4=128, dropout_rate=0.2):
        super(AttentionNCF, self).__init__()
        # save the (hyper) parameters needed to construct this object when saving model
        self.kwargs = {'item_dim': item_dim, 'item_emb': item_emb, 'user_emb': user_emb, 'att_dense': att_dense,
                       'dense1': dense1, 'dense2': dense2, 'dense3': dense3, 'dense4': dense4}
        # layers
        self.ItemEmbeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb),
            nn.ReLU()
        )
        self.UserEmbeddings = nn.Sequential(
            nn.Linear(item_dim, user_emb),
            nn.ReLU()
        )
        if att_dense is not None:
            self.AttentionNet = nn.Sequential(
                nn.Linear(2*item_emb, att_dense),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(att_dense, 1)
            )
        else:
            self.AttentionNet = nn.Sequential(
                nn.Linear(2 * item_emb, 1)
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
            nn.Linear(dense3, dense4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense4, 1)
        )

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def forward(self, candidate_items, rated_items, user_matrix, candidate_names=None, rated_names=None):
        I = rated_items.shape[0]      # == user_matrix.shape[1]
        B = candidate_items.shape[0]  # == user_matrix.shape[0]

        # pass through item embeddings layer
        candidate_item_embeddings = self.ItemEmbeddings(candidate_items)

        # TODO:  Use detach or not?
        rated_emb = self.ItemEmbeddings(rated_items)  # .detach()

        # attention on rated items
        """ Note: the one that interleaves matters! I think this works correctly into (B, I) shape 
        because the first I elements contain all different rated items and they become the first row of length I """
        attNetInput = torch.cat((candidate_item_embeddings.repeat_interleave(I, dim=0), rated_emb.repeat(B, 1)), dim=1)
        attention_scores = self.AttentionNet(attNetInput).view(B, I)
        # mask unrated items per user (!) - otherwise there may be high weights on 0 entries
        attention_scores[user_matrix == 0.0] = -float('inf')    # so that softmax gives this a 0 attention weight
        # pass through softmax
        attention_scores = F.softmax(attention_scores, dim=1)   # (B, I)
        attention_scores = attention_scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # will get NaNs if a user has 0 ratings. Replace those with 0

        # visualize attention
        if candidate_names is not None and rated_names is not None:
            visualize_attention(attention_scores.to('cpu'), user_matrix.to('cpu'), candidate_names, rated_names.values)

        # aggregate item features based on ratings and attention weights
        attended_user_matrix = torch.mul(attention_scores, user_matrix)
        user_estimated_features = torch.matmul(attended_user_matrix, rated_items)

        # pass through user embeddings layer
        user_embeddings = self.UserEmbeddings(user_estimated_features)   # TODO: or use embeddings here as well?

        # combine
        combined = torch.cat((candidate_item_embeddings, user_embeddings), dim=1)

        # MLP part
        out = self.MLP(combined)

        return out
