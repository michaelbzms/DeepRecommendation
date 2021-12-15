from abc import abstractmethod
from torch.utils.data import Dataset

from gnns.models.GNN_NCF import GNN_NCF


class GNN_Dataset(Dataset):
    """
    Use __getitem__() to return batches of TODO: decide
    """

    @abstractmethod
    def __getitem__(self, item):
        raise Exception('Not Implemented')

    @staticmethod
    def use_collate():
        return None  # no custom collate needed  TODO

    @staticmethod
    def forward(model: GNN_NCF, batch, device):
        # get the input matrices and the target
        item_vec, user_vec, y_batch = batch
        # forward model
        out = model(item_vec.float().to(device), user_vec.float().to(device))
        return out, y_batch
