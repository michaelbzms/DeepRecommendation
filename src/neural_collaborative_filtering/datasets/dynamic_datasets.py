from neural_collaborative_filtering.content_providers import DynamicContentProvider
from neural_collaborative_filtering.datasets.base import PointwiseDataset, RankingDataset
from neural_collaborative_filtering.models.base import NCF


class DynamicPointwiseDataset(PointwiseDataset):
    """
    Use this dataset if user vector input not fixed but instead we want to construct it from item vectors for
    items the user has interacted with. For point-wise learning.

    Combine __getitem__() and possibly a custom collate_fn to return for each batch in a data loader:
    > candidate_items_batch: (B, F)  B items with their features
    > rated_items_features: (I, F) I rated items with their features
    > user_matrix: (B, I) a subarray (not exactly) of the utility matrix with the (normalized) ratings of B users on I items.
    The order must match rated_items_feature's order on I axis.
    """
    def __init__(self, file: str, dynamic_provider: DynamicContentProvider):
        super().__init__(file)
        self.dynamic_provider = dynamic_provider

    def use_collate(self):
        return lambda batch: self.dynamic_provider.collate_interacted_items(batch, for_ranking=False)

    @staticmethod
    def do_forward(model: NCF, batch, device, return_attention_weights=False):
        # get the input matrices and the target
        candidate_items_IDs, rated_items_IDs, candidate_items, rated_items, user_matrix, y_batch = batch
        # forward model
        if return_attention_weights:
            out, att_weights = model(candidate_items.float().to(device),
                                     rated_items.float().to(device),
                                     user_matrix.float().to(device),
                                     return_attention_weights=True)
            return out, y_batch, candidate_items_IDs, rated_items_IDs, att_weights, user_matrix
        else:
            out = model(candidate_items.float().to(device),
                        rated_items.float().to(device),
                        user_matrix.float().to(device),
                        return_attention_weights=False)
            return out, y_batch


class DynamicRankingDataset(RankingDataset):
    """
    Same but for pairwise learning.
    """
    def __init__(self, ranking_file: str, dynamic_provider: DynamicContentProvider):
        super().__init__(ranking_file)
        self.dynamic_provider = dynamic_provider

    def use_collate(self):
        return lambda batch: self.dynamic_provider.collate_interacted_items(batch, for_ranking=True)

    @staticmethod
    def do_forward(model: NCF, batch, device):
        # get the input matrices and the target
        candidate_items_IDs, rated_items_IDs, candidate_items1, rated_items, user_matrix, candidate_items2 = batch
        # forward model
        out1 = model(candidate_items1.float().to(device), rated_items.float().to(device), user_matrix.float().to(device))
        out2 = model(candidate_items2.float().to(device), rated_items.float().to(device), user_matrix.float().to(device))
        return out1, out2
