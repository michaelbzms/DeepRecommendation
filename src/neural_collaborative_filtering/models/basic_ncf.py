import torch
from torch import nn

from neural_collaborative_filtering.datasets.fixed_datasets import FixedPointwiseDataset, FixedRankingDataset
from neural_collaborative_filtering.models.base import NCF
from neural_collaborative_filtering.util import build_MLP_layers


class BasicNCF(NCF):
    def __init__(self, item_dim, user_dim, dropout_rate=0.2,
                 item_emb=256, user_emb=256, mlp_dense_layers=None):
        super(BasicNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [256, 128]

        # save the (hyper) parameters needed to construct this object when saving model
        self.kwargs = {
            'item_dim': item_dim,
            'user_dim': user_dim,
            'item_emb': item_emb,
            'user_emb': user_emb,
            'mlp_dense_layers': mlp_dense_layers,
            'dropout_rate': dropout_rate
        }

        # embedding layers
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb)
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(user_dim, user_emb)
        )

        # build MLP according to params
        self.MLP = build_MLP_layers(item_emb + user_emb, mlp_dense_layers, dropout_rate=dropout_rate)

    def forward(self, X_user, X_item):
        user_emb = self.user_embeddings(X_user)
        item_emb = self.item_embeddings(X_item)
        combined = torch.cat((user_emb, item_emb), dim=1)
        out = self.MLP(combined)
        return out

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, FixedPointwiseDataset) or issubclass(dataset_class, FixedRankingDataset)
