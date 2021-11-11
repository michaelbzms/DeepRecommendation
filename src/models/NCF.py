from abc import abstractmethod
import numpy as np
from torch import nn
import torch


class NCF(nn.Module):
    def __init__(self):
        super(NCF, self).__init__()

    @abstractmethod
    def forward(self, candidate_items_batch: np.array, rated_items_features: np.array, user_matrix: np.array):
        """
        Executes a forward pass of the network for a batch of B samples which are known cells in the utility matrix.
        :param candidate_items_batch: (B, F)  B items with their features
        :param rated_items_features: (I, F) I rated items with their features
        :param user_matrix: (B, I) a subarray of the utility matrix with the (normalized?) ratings of B users on I items
        :return:
        """
        raise Exception('Not Implemented')

    @abstractmethod
    def get_model_parameters(self) -> dict[str]:
        raise Exception('Not Implemented')

    def save_model(self, file):
        torch.save([self.state_dict(), self.get_model_parameters()], file)


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
