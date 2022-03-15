from abc import abstractmethod

from neural_collaborative_filtering.datasets.base import NCF_dataset
from neural_collaborative_filtering.models.base import GNN_NCF


class GNN_Dataset(NCF_dataset):
    """
    Use __getitem__() to return batches of (user_index, item_index, target rating)
    """
    @abstractmethod
    def __getitem__(self, item):
        """ return (userId, itemId, target) or use custom collate to make batches of such """
        raise Exception('Not Implemented')

    @abstractmethod
    def get_graph(self, device):
        raise Exception('Not Implemented')

    @staticmethod
    def do_forward(model: GNN_NCF, batch, device, graph, *args):
        """ expects samples of (userId, itemId, target) and a graph to pass on to the model """
        # get the input matrices and the target
        userIds, itemIds, y_batch = batch
        # forward model
        out = model(graph, userIds.float().to(device), itemIds.float().to(device), *args)
        return out, y_batch

    @abstractmethod
    def get_class_counts(self):
        raise Exception('Not Implemented')
