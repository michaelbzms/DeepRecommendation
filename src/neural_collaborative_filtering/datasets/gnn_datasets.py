from torch.utils.data import Dataset

from neural_collaborative_filtering.models.base import GNN_NCF


class GraphPointwiseDataset(Dataset):
    """
    Use __getitem__() to return batches of (user_index, item_index, target rating)
    """

    def __getitem__(self, item):
        # TODO return (user index, item index, target)
        pass
        # data = self.set.iloc[item]
        # return int(self.all_users_index[data['userId']]), int(self.all_items_index[data['movieId']]), float(data['rating'])

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
