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

    @abstractmethod
    def get_graph(self):
        raise Exception('Not Implemented')

    @staticmethod
    def use_collate():
        return None  # no custom collate needed  TODO

    @staticmethod
    def forward(model: GNN_NCF, graph, batch, device):
        # get the input matrices and the target
        userIds, itemIds, y_batch = batch
        # forward model
        out = model(graph, userIds.float().to(device), itemIds.float().to(device))
        return out, y_batch
