import torch
from torch import nn

from neural_collaborative_filtering.datasets.fixed_datasets import FixedPointwiseDataset, FixedRankingDataset
from neural_collaborative_filtering.models.base import NCF


class MF(NCF):
    def __init__(self, item_dim, user_dim, item_emb=128, user_emb=128):
        super(MF, self).__init__()

        # save the (hyper) parameters needed to construct this object when saving model
        self.kwargs = {
            'item_dim': item_dim,
            'user_dim': user_dim,
            'item_emb': item_emb,
            'user_emb': user_emb,
        }

        # embedding layers
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb)
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(user_dim, user_emb)
        )

    def forward(self, X_user, X_item):
        user_emb = self.user_embeddings(X_user)
        item_emb = self.item_embeddings(X_item)
        out = torch.bmm(user_emb.unsqueeze(1), item_emb.unsqueeze(2)).view(-1, 1)  # dot product
        return out

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, FixedPointwiseDataset) or issubclass(dataset_class, FixedRankingDataset)
