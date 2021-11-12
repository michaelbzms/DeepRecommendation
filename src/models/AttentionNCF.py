import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from models.NCF import NCF
from plots import visualize_attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionNCF(NCF):
    def __init__(self, item_dim, item_emb=256, user_emb=256, att_dense=None,
                 mlp_dense_layers=None, dropout_rate=0.2):
        super(AttentionNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [512, 256, 128]           # default
        # save the (hyper) parameters needed to construct this object when saving model
        self.kwargs = {'item_dim': item_dim, 'item_emb': item_emb, 'user_emb': user_emb, 'att_dense': att_dense,
                       'mlp_dense_layers': mlp_dense_layers}
        # embedding layers
        self.ItemEmbeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb),
            nn.ReLU()
        )
        self.UserEmbeddings = nn.Sequential(
            nn.Linear(item_dim, user_emb),
            nn.ReLU()
        )
        # build attention network
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
        # Build MLP according to
        mlp_dense_layers = list(mlp_dense_layers)    # make mutable if it's not
        mlp_dense_layers.append(1)                   # the output layer
        num_layers = len(mlp_dense_layers)
        mlp_layers = [nn.Linear(item_emb + user_emb, mlp_dense_layers[0])]   # input layer
        for i in range(1, num_layers):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            mlp_layers.append(nn.Linear(mlp_dense_layers[i - 1], mlp_dense_layers[i]))
        self.MLP = nn.Sequential(*mlp_layers)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def forward(self, candidate_items, rated_items, user_matrix, candidate_names=None, rated_names=None, att_stats=None, visualize=False):
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
        if visualize and candidate_names is not None and rated_names is not None:
            visualize_attention(attention_scores.to('cpu').detach(), user_matrix.to('cpu').detach(), candidate_names, rated_names.values)

        # keep stats for attention weights and items
        if att_stats is not None and candidate_names is not None and rated_names is not None:
            att_scores_np = attention_scores.to('cpu').detach().numpy()
            counts = np.zeros((B, I), dtype=np.int)
            counts[user_matrix.to('cpu').detach() != 0.0] = 1
            # Old attempt: att_scores_df = pd.DataFrame(index=candidate_names, columns=rated_names['primaryTitle'], data=att_scores_np)
            for i in range(len(candidate_names)):
                att_stats['sum'].loc[candidate_names[i]] += att_scores_np[i, :]
                att_stats['count'].loc[candidate_names[i]] += counts[i, :]

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
