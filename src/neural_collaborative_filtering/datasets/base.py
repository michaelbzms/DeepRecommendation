import torch
from torch.utils.data import Dataset
from torch import nn
import pandas as pd


class PointwiseDataset(Dataset):
    def __init__(self, file):
        # expects to read (user, item, rating) triplets
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')
        # use MSE loss
        self.loss_fn = nn.MSELoss(reduction='sum')

    def __getitem__(self, item):
        # return (user ID, item ID, rating) triplets
        data = self.samples.iloc[item]
        return data['userId'], data['movieId'], data['rating']

    def calculate_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.view(-1, 1).float())


class RankingDataset(Dataset):
    def __init__(self, ranking_file):
        # expects to read (user, item1, item2) triplets where item1 > item2 for user
        self.samples: pd.DataFrame = pd.read_csv(ranking_file + '.csv')
        # use BPR loss for ranking
        self.loss_fn = BPR_loss

    def __getitem__(self, item):
        # return (user ID, item1 ID, item2 ID) triplets
        data = self.samples.iloc[item]
        return data['userId'], data['movieId1'], data['movieId2']

    def calculate_loss(self, out_pos, out_neg):
        return self.loss_fn(out_pos, out_neg)


def BPR_loss(out_pos, out_neg):
    return torch.sum(-torch.log(torch.sigmoid(out_pos - out_neg)))
