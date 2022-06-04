from abc import abstractmethod
from torch import nn
import torch


class NCF(nn.Module):
    def __init__(self):
        super(NCF, self).__init__()

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def get_model_parameters(self) -> dict[str]:
        raise NotImplementedError

    def save_model(self, file):
        torch.save([self.state_dict(), self.get_model_parameters()], file)

    @abstractmethod
    def is_dataset_compatible(self, dataset_class):
        raise NotImplementedError

    def important_hypeparams(self) -> str:
        return ''


class GNN_NCF(NCF):
    def __init__(self):
        super(GNN_NCF, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Executes a forward pass of the network for a batch of B samples which are known cells in the utility matrix.
        :param   graph: PyG's Data graph with node features and edges
        :param userIds: (B, 1) user IDs in batch
        :param itemIds: (B, 1) item IDs in batch
        :return:
        """
        raise NotImplementedError
