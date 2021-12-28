from abc import abstractmethod
from torch import nn
import torch


class NCF(nn.Module):
    def __init__(self):
        super(NCF, self).__init__()

    @abstractmethod
    def forward(self, *args):
        """
        Executes a forward pass of the network for a batch of B samples which are known cells in the utility matrix.
        Parameters for non one-hot input models
        :param candidate_items_batch: (B, F)  B items with their features
        :param rated_items_features: (I, F) I rated items with their features
        :param user_matrix: (B, I) a subarray of the utility matrix with the (normalized?) ratings of B users on I items.
            The order must match rated_items_feature's order on I axis.
        :return:
        """
        raise Exception('Not Implemented')

    @abstractmethod
    def get_model_parameters(self) -> dict[str]:
        raise Exception('Not Implemented')

    def save_model(self, file):
        torch.save([self.state_dict(), self.get_model_parameters()], file)

    @abstractmethod
    def is_dataset_compatible(self, dataset_class):
        raise Exception('Not Implemented')


class GNN_NCF(NCF):
    def __init__(self):
        super(GNN_NCF, self).__init__()

    @abstractmethod
    def forward(self, *args):
        """
        Executes a forward pass of the network for a batch of B samples which are known cells in the utility matrix.
        :param   graph: PyG's Data graph with node features and edges
        :param userIds: (B, 1) user IDs in batch
        :param itemIds: (B, 1) item IDs in batch
        :return:
        """
        raise Exception('Not Implemented')
