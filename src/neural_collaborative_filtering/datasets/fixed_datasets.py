import torch
from torch.utils.data import Dataset
import pandas as pd

from neural_collaborative_filtering.content_providers import ContentProvider
from neural_collaborative_filtering.models.base import NCF


class PointwiseDataset(Dataset):
    """ Base template and functionality for point-wise learning and evaluation """

    def __init__(self, file: str, content_provider: ContentProvider):
        # expects to read (user, item, rating) triplets
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')
        self.content_provider = content_provider

    def __getitem__(self, item):
        # return (user ID, item ID, rating) triplets
        data = self.samples.iloc[item]
        user_vec = self.content_provider.get_user_profile(userID=data['userId'])
        item_vec = self.content_provider.get_item_profile(itemID=data['movieId'])
        return torch.FloatTensor(user_vec), torch.FloatTensor(item_vec), float(data['rating'])

    def __len__(self):
        return len(self.samples)

    def get_graph(self, device):    # TODO: remove?
        return None

    def use_collate(self):
        return None

    @staticmethod
    def do_forward(model: NCF, batch, device, *args):
        # get the input matrices and the target
        user_vec, item_vec, y_batch = batch
        # forward model
        out = model(user_vec.float().to(device), item_vec.float().to(device))
        # TODO: loss here
        return out, y_batch


class RankingDataset(Dataset):
    """ Base template and functionality for pair-wise learning and evaluation """

    def __init__(self, ranking_file: str, content_provider: ContentProvider):
        # expects to read (user, item1, item2) triplets where item1 > item2 for user
        self.samples: pd.DataFrame = pd.read_csv(ranking_file + '.csv')
        self.content_provider = content_provider

    def __getitem__(self, item):
        # return (user ID, item1 ID, item2 ID) triplets
        data = self.samples.iloc[item]
        user_vec = self.content_provider.get_user_profile(userID=data['userId'])
        item1_vec = self.content_provider.get_item_profile(itemID=data['movieId1'])
        item2_vec = self.content_provider.get_item_profile(itemID=data['movieId2'])
        return torch.FloatTensor(user_vec), torch.FloatTensor(item1_vec), torch.FloatTensor(item2_vec)

    def __len__(self):
        return len(self.samples)

    def get_graph(self, device):  # TODO: remove?
        return None

    def use_collate(self):
        return None

    @staticmethod
    def do_forward(model: NCF, batch, device, *args):
        # get the input matrices and the target
        user_vec, item1_vec, item2_vec = batch
        # forward model
        out1 = model(user_vec.float().to(device), item1_vec.float().to(device))
        out2 = model(user_vec.float().to(device), item2_vec.float().to(device))
        # TODO: loss here
        return out1, out2
