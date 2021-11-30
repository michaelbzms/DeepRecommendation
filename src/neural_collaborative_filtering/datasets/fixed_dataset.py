from abc import abstractmethod

from torch.utils.data import Dataset

from neural_collaborative_filtering.models.NCF import NCF


class FixedDataset(Dataset):
    """
    Use __getitem__() to return batches of (item_vec, user_vec, y) samples
    that are stacked by the default collate_fn()
    > item_vec could be one-hot encoded as in NCF or fixed item features or fixed item embeddings
    > user_vec could be one-hot encoded as in NCF or fixed user features or fixed user embeddings
    """

    @abstractmethod
    def __getitem__(self, item):
        raise Exception('Not Implemented')

    @staticmethod
    def use_collate():
        return None  # no custom collate needed

    @staticmethod
    def get_item_feature_dim():   # aka F
        raise Exception('Not Implemented')

    @staticmethod
    def forward(model: NCF, batch, device):
        # get the input matrices and the target
        item_vec, user_vec, y_batch = batch
        # forward model
        out = model(item_vec.float().to(device), user_vec.float().to(device))
        return out, y_batch
