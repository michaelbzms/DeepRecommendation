from neural_collaborative_filtering.datasets.base import PointwiseDataset
from neural_collaborative_filtering.models.base import GNN_NCF


class GraphPointwiseDataset(PointwiseDataset):
    """
    Use __getitem__() to return batches of (user_index, item_index, target rating)
    """
    def __init__(self, file: str):
        super().__init__(file)

    def __getitem__(self, item):
        # TODO return (user index, item index, target)
        pass

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

    def get_class_counts(self):
        raise Exception('Not Implemented')
