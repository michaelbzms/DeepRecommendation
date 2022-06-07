import sys

import torch
from torch import nn
import torch.nn.functional as F

from neural_collaborative_filtering.datasets.dynamic_datasets import DynamicPointwiseDataset, DynamicRankingDataset
from neural_collaborative_filtering.models.base import NCF
from neural_collaborative_filtering.util import build_MLP_layers


class AdvancedNCF(NCF):
    def __init__(self, item_dim, item_emb=128, user_emb=128, mlp_dense_layers=None, dropout_rate=0.2):
        super(AdvancedNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [256, 128]         # default
        self.kwargs = {'item_dim': item_dim, 'item_emb': item_emb, 'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers, 'dropout_rate': dropout_rate}
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb)
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(item_emb, user_emb)         # TODO: this inputs item_emb
        )
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, DynamicPointwiseDataset) or issubclass(dataset_class, DynamicRankingDataset)

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


class AttentionNCF(NCF):
    def __init__(self, item_dim, item_emb=128, user_emb=128, att_dense=None,
                 mlp_dense_layers=None, use_cos_sim_instead=False, dropout_rate=0.2,
                 message_dropout=None):
        super(AttentionNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [256, 128]           # default
        # save the (hyper) parameters needed to construct this object when saving model
        self.kwargs = {
            'item_dim': item_dim,
            'item_emb': item_emb,
            'user_emb': user_emb,
            'att_dense': att_dense,
            'mlp_dense_layers': mlp_dense_layers,
            'dropout_rate': dropout_rate,
            'use_cos_sim_instead': use_cos_sim_instead,
            'message_dropout': message_dropout
        }
        self.use_cos_sim_instead = use_cos_sim_instead
        self.message_dropout = message_dropout

        # embedding layers
        self.ItemEmbeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb)
        )

        self.UserEmbeddings = nn.Sequential(
            nn.Linear(item_dim, user_emb)
        )

        # build attention network if not using cosine similarity
        if not self.use_cos_sim_instead:
            if att_dense is not None:
                self.att_dense = att_dense
                self.AttentionNet = nn.Sequential(
                    nn.Linear(2 * item_emb, att_dense),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(att_dense, 1)
                )
            else:
                self.att_dense = 0
                self.AttentionNet = nn.Sequential(
                    nn.Linear(2 * item_emb, 1)
                )

        # Build MLP according to params
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def important_hypeparams(self) -> str:
        return '_cosine' if self.use_cos_sim_instead else f'_attNet{self.att_dense}'

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, DynamicPointwiseDataset) or issubclass(dataset_class, DynamicRankingDataset)

    def forward(self, candidate_items, rated_items, user_matrix, return_attention_weights=False):
        I = rated_items.shape[0]      # == user_matrix.shape[1]
        B = candidate_items.shape[0]  # == user_matrix.shape[0]

        # pass through item embeddings layer
        candidate_item_embeddings = self.ItemEmbeddings(candidate_items)

        # Note:  Use detach or not?  -> using it gives slightly worse results
        rated_emb = self.ItemEmbeddings(rated_items)

        """ Attention on rated items
        Note: the one that interleaves matters! I think this works correctly into (B, I) shape 
        because the first I elements contain all different rated items and they become the first row of length I 
        """

        # create attention input pairs (needed for masking also)
        candidate_interleaved_full = candidate_item_embeddings.repeat_interleave(I, dim=0)
        rated_interleaved_full = rated_emb.repeat(B, 1)

        # Optimization: forward AttentionNet / do cos similarity only on valid item pairs by pre-filtering
        # based on unrated items. We need to mask unrated items per user as
        # otherwise there will be attention given on 0 entries of the user matrix.
        candidate_interleaved = candidate_interleaved_full.view(B, I, -1)[user_matrix != 0]
        rated_interleaved = rated_interleaved_full.view(B, I, -1)[user_matrix != 0]

        if self.use_cos_sim_instead:
            def cos_sim(a: torch.Tensor, b: torch.Tensor, only_dot_product=False):
                if len(a.shape) == 1: a = a.unsqueeze(0)
                if len(b.shape) == 1: b = b.unsqueeze(0)
                if not only_dot_product:  # only dot product is not good, too restrictive maybe
                    a = torch.nn.functional.normalize(a, p=2, dim=1)
                    b = torch.nn.functional.normalize(b, p=2, dim=1)
                # dot product of (optionally normalized) batched vectors
                return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).view(-1)

            # calculate all similarities
            attOut = cos_sim(candidate_interleaved, rated_interleaved)
        else:
            # concatenate two input embeddings
            attNetInput = torch.cat((candidate_interleaved, rated_interleaved), dim=1)

            # forward AttentionNet only to valid candidate - rated item combos (optimization)
            attOut = self.AttentionNet(attNetInput).view(-1)

        # create attention scores matrix
        attention_scores = -float('inf') * torch.ones((B, I), dtype=torch.float32, device=attOut.device)

        # perform message dropout by randomly setting some att weights to zero
        if self.training and self.message_dropout is not None:
            # randomly zero some
            F.dropout(attOut, p=self.message_dropout, training=self.training, inplace=True)
            # replace zeroes with -inf because of softmax later
            attOut[attOut == 0] = -float('inf')

        attention_scores[user_matrix != 0.0] = attOut

        # mask item we are trying to rate for each user if we are training so that the network does not learn to overfit
        if self.training:
            try:
                # TODO: atol is risky because we might accidentally get others or miss valid ones in which case view() will throw an error..
                _mask = torch.isclose(candidate_interleaved_full, rated_interleaved_full, atol=1e-5).all(dim=1).view(B, I)
                if _mask.shape[0] != B:    # should not happen for a reasonable atol value
                    print("Warning: Something went wrong!", file=sys.stderr)
                # perform mask
                attention_scores[_mask] = -float('inf')
            except:
                print("Warning: Could not calculate training mask. Ignoring masking this time.", file=sys.stderr)  # should not happen

        # pass through softmax
        attention_scores = F.softmax(attention_scores, dim=1)   # (B, I)
        attention_scores = attention_scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # will get NaNs if a user has 0 ratings. Replace those with 0

        # aggregate item features based on ratings and attention weights to build user profiles
        attended_user_matrix = torch.mul(attention_scores, user_matrix)
        user_estimated_features = torch.matmul(attended_user_matrix, rated_items)   # TODO: or aggregate item embeddings? -> makes user part cheaper

        # pass through user embeddings layer
        user_embeddings = self.UserEmbeddings(user_estimated_features)

        # combine
        combined = torch.cat((candidate_item_embeddings, user_embeddings), dim=1)

        # MLP part
        out = self.MLP(combined)

        return out if not return_attention_weights else (out, attention_scores)
