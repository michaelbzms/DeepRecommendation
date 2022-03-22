import torch
from torch.utils.data import Dataset
from torch import nn


class PointwiseDataset(Dataset):
    def __init__(self):
        # use MSE loss
        self.loss_fn = nn.MSELoss(reduction='sum')

    def __getitem__(self, item):
        raise NotImplementedError

    def calculate_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.view(-1, 1).float())


class RankingDataset(Dataset):
    def __init__(self):
        # use BPR loss for ranking
        self.loss_fn = BPR_loss

    def __getitem__(self, item):
        raise NotImplementedError

    def calculate_loss(self, out_pos, out_neg):
        return self.loss_fn(out_pos, out_neg)


def BPR_loss(out_pos, out_neg):
    return torch.sum(-torch.log(torch.sigmoid(out_pos - out_neg)))
