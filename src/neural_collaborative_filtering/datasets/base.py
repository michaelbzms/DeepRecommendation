from abc import abstractmethod
from torch.utils.data import Dataset


class NCF_dataset(Dataset):

    @abstractmethod
    def __getitem__(self, item):
        raise Exception('Not Implemented')

    def get_graph(self, device):
        return None

    @staticmethod
    def use_collate():
        return None

    @staticmethod
    def do_forward(*args):
        """ Define how a model compatible with this dataset should perform a forward pass """
        raise Exception('Not Implemented!')
