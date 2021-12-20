from abc import abstractmethod
from torch.utils.data import Dataset

from gnns.models.GNN import GNN_NCF


class GNN_Dataset(Dataset):
    """
    Use __getitem__() to return batches of TODO: decide
    """
    @abstractmethod
    def __getitem__(self, item):
        """ return (userId, itemId, target) or use custom collate to make batches of such """
        raise Exception('Not Implemented')

    @abstractmethod
    def get_graph(self):
        raise Exception('Not Implemented')

    @staticmethod
    def use_collate():
        return None  # no custom collate needed  TODO

    @staticmethod
    def forward(model: GNN_NCF, graph, batch, device, *args):
        """ expects samples of  (userId, itemId, target) and a graph to pass on to the model """
        # get the input matrices and the target
        userIds, itemIds, y_batch = batch
        # forward model
        out = model(graph, userIds.float().to(device), itemIds.float().to(device), *args)
        return out, y_batch

    @abstractmethod
    def get_class_counts(self):
        raise Exception('Not Implemented')

