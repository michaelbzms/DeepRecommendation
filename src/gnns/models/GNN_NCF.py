from abc import abstractmethod
from torch import nn
import torch


class GNN_NCF(nn.Module):
    def __init__(self):
        super(GNN_NCF, self).__init__()

    @abstractmethod
    def forward(self, *args):
        """
        Executes a forward pass of the network for a batch of B samples which are known cells in the utility matrix.
        Example parameters for USE_FEATURES == True
        :param   graph: PyG's Data graph with node features and edges
        :param userIds: (B, 1) user IDs in batch
        :param itemIds: (B, 1) item IDs in batch
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


def load_model_state_and_params(file, ModelClass=None):
    state, kwargs = torch.load(file)
    if ModelClass is None:
        return state, kwargs
    else:
        model = ModelClass(**kwargs)
        model.load_state_dict(state)
        return model


def load_model(file, ModelClass):
    return load_model_state_and_params(file, ModelClass)
