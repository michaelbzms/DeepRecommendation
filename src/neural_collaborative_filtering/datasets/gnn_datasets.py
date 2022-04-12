from neural_collaborative_filtering.content_providers import ContentProvider
from neural_collaborative_filtering.datasets.base import PointwiseDataset
from neural_collaborative_filtering.models.base import GNN_NCF


class GraphPointwiseDataset(PointwiseDataset):
    """
    Use __getitem__() to return batches of (user_index, item_index, target rating)
    """
    def __init__(self, file: str, content_provider: ContentProvider):
        super().__init__(file)
        self.cp = content_provider
        # TODO: create graph from content provider and file

    def __getitem__(self, item):
        # TODO return (user node index in graph, item node index in graph, target)
        userID, itemID, target = super().__getitem__(item)
        # todo from user & item IDs get their node IDs in the graph
        pass

    def get_graph(self, device):
        raise NotImplementedError

    @staticmethod
    def do_forward(model: GNN_NCF, batch, device, graph, *args):
        """ expects samples of (userId, itemId, target) and a graph to pass on to the model """
        # get the input matrices and the target
        userIds, itemIds, y_batch = batch
        # forward model
        out = model(graph, userIds.float().to(device), itemIds.float().to(device), *args)
        return out, y_batch

    def get_class_counts(self):
        raise NotImplementedError
