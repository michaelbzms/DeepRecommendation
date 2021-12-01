from abc import abstractmethod

from torch.utils.data import Dataset

from neural_collaborative_filtering.models.NCF import NCF


class DynamicDataset(Dataset):
    """
    Combine __getitem__() and possibly a custom collate_fn to return for each batch in a data loader:
    > candidate_items_batch: (B, F)  B items with their features
    > rated_items_features: (I, F) I rated items with their features
    > user_matrix: (B, I) a subarray (not exactly) of the utility matrix with the (normalized) ratings of B users on I items.
    The order must match rated_items_feature's order on I axis.
    """

    @abstractmethod
    def __getitem__(self, item):
        raise Exception('Not Implemented')

    @staticmethod
    def use_collate():
        # Return collate_fn() function to use. If None will default to stacking samples from __getitem__()
        raise Exception('Not Implemented')

    @staticmethod
    def get_item_feature_dim():   # aka F
        raise Exception('Not Implemented')

    @staticmethod
    def get_number_of_items():    # aka I
        raise Exception('Not Implemented')

    @staticmethod
    def get_sorted_item_names():
        raise Exception('Not Implemented')

    @staticmethod
    def forward(model: NCF, batch, device):
        # get the input matrices and the target
        candidate_items, rated_items, user_matrix, y_batch = batch
        # forward model
        out = model(candidate_items.float().to(device), rated_items.float().to(device), user_matrix.float().to(device))
        return out, y_batch