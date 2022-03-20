import torch
from torch import nn

from neural_collaborative_filtering.datasets.base import PointwiseDataset, RankingDataset
from neural_collaborative_filtering.models.base import NCF
from neural_collaborative_filtering.util import build_MLP_layers


class BasicNCF(NCF):
    def __init__(self, item_dim, user_dim, dropout_rate=0.2,
                 item_emb=256, user_emb=256, mlp_dense_layers=None):
        super(BasicNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [256, 128]
        # save the (hyper) parameters needed to construct this object when saving model
        self.kwargs = {'item_dim': item_dim, 'user_dim': user_dim,
                       'item_emb': item_emb, 'user_emb': user_emb,
                       'mlp_dense_layers': mlp_dense_layers}
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb),
            nn.ReLU()
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(user_dim, user_emb),
            nn.ReLU()
        )
        # Build MLP according to params
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
        return issubclass(dataset_class, PointwiseDataset) or issubclass(dataset_class, RankingDataset)


class BasicMultimodalNCF(NCF):
    def __init__(self, item_dim, user_dim, dropout_rate=0.2,
                 item_emb=256, user_emb=256, att_dense=128,
                 num_heads=1, mlp_dense_layers=None):
        super(BasicMultimodalNCF, self).__init__()
        if mlp_dense_layers is None:
            mlp_dense_layers = [256, 128]
        # save the (hyper) parameters needed to construct this object when saving model
        self.kwargs = {'item_dim': item_dim, 'user_dim': user_dim,
                       'item_emb': item_emb, 'user_emb': user_emb,
                       'att_dense': att_dense, 'num_heads': num_heads,
                       'mlp_dense_layers': mlp_dense_layers}
        self.item_embeddings = nn.Sequential(
            nn.Linear(item_dim, item_emb),
            nn.LeakyReLU()
        )
        self.user_embeddings = nn.Sequential(
            nn.Linear(user_dim, user_emb),
            nn.LeakyReLU()
        )

        self.W_query = nn.Linear(item_emb, att_dense)
        self.W_key = nn.Linear(user_emb, att_dense)
        self.W_value = nn.Linear(user_emb, att_dense)
        self.MultiHeadAttLayer = nn.MultiheadAttention(1, num_heads=num_heads, batch_first=True)   # TODO: dropout?

        self.W_query2 = nn.Linear(user_emb, att_dense)
        self.W_key2 = nn.Linear(item_emb, att_dense)
        self.W_value2 = nn.Linear(item_emb, att_dense)
        self.MultiHeadAttLayer2 = nn.MultiheadAttention(1, num_heads=num_heads, batch_first=True)  # TODO: dropout?

        # Build MLP according to params
        self.MLP = build_MLP_layers(2 * att_dense, mlp_dense_layers, dropout_rate=dropout_rate)

    def forward(self, X_user, X_item):
        item_emb = self.item_embeddings(X_item)
        user_emb = self.user_embeddings(X_user)

        # item --> user attention
        Q1 = torch.unsqueeze(self.W_query(item_emb), 2)
        K1 = torch.unsqueeze(self.W_key(user_emb), 2)
        V1 = torch.unsqueeze(self.W_value(user_emb), 2)
        att_emb1, att_weights1 = self.MultiHeadAttLayer(Q1, K1, V1)
        att_emb1 = torch.squeeze(att_emb1, 2)

        # user --> item attention
        Q2 = torch.unsqueeze(self.W_query2(user_emb), 2)
        K2 = torch.unsqueeze(self.W_key2(item_emb), 2)
        V2 = torch.unsqueeze(self.W_value2(item_emb), 2)
        att_emb2, att_weights2 = self.MultiHeadAttLayer(Q2, K2, V2)
        att_emb2 = torch.squeeze(att_emb2, 2)

        combined = torch.cat((att_emb1, att_emb2), dim=1)

        out = self.MLP(combined)

        return out

    def get_model_parameters(self) -> dict[str]:
        return self.kwargs

    def is_dataset_compatible(self, dataset_class):
        return issubclass(dataset_class, FixedDataset)
