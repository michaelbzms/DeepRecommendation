import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from neural_collaborative_filtering.datasets.dynamic_dataset import DynamicDataset
from neural_collaborative_filtering.models.base import NCF
from neural_collaborative_filtering.plots import visualize_attention
from neural_collaborative_filtering.util import build_MLP_layers


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

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, DynamicDataset)

    def forward(self, *, candidate_items, item_matrix, user_matrix):
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


class AttentionNCF(NCF):
    def __init__(self, item_dim, item_emb=128, user_emb=128, att_dense=None,
                 mlp_dense_layers=None, dropout_rate=0.2):
        super(AttentionNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [256, 128]           # default
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
                nn.Linear(2 * item_dim, att_dense),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(att_dense, 1)
            )
        else:
            self.AttentionNet = nn.Sequential(
                nn.Linear(2 * item_dim, 1)
            )
        # Build MLP according to params
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, DynamicDataset)

    def forward(self, *, candidate_items, rated_items, user_matrix, candidate_names=None, rated_names=None, att_stats=None, visualize=False):
        I = rated_items.shape[0]      # == user_matrix.shape[1]
        B = candidate_items.shape[0]  # == user_matrix.shape[0]

        # pass through item embeddings layer
        candidate_item_embeddings = self.ItemEmbeddings(candidate_items)

        # Note:  Use detach or not?  -> using it gives slightly worse results
        # rated_emb = self.ItemEmbeddings(rated_items)

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
        user_embeddings = self.UserEmbeddings(user_estimated_features)

        # combine
        combined = torch.cat((candidate_item_embeddings, user_embeddings), dim=1)

        # MLP part
        out = self.MLP(combined)

        return out
