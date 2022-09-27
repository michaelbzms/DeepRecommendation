import torch
from torch.utils.data import Dataset
from torch import nn, softmax
import pandas as pd
import numpy as np


class PointwiseDataset(Dataset):
    """
    Base template and functionality for point-wise learning and evaluation.
    """
    def __init__(self, file, use_bce_loss=False):
        # expects to read (user, item, rating) triplets
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')
        self.use_bce_loss = use_bce_loss
        if use_bce_loss:
            # use BCE loss with logits (soft targets)
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            # use MSE loss
            self.loss_fn = nn.MSELoss(reduction='sum')

    def __getitem__(self, item):
        # return (user ID, item ID, rating) triplets
        data = self.samples.iloc[item]
        return data['userId'], data['movieId'], data['rating'] / 5.0 if self.use_bce_loss else data['rating']

    def __len__(self):
        return len(self.samples)

    def calculate_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.view(-1, 1).float())

    def get_graph(self, device):   # default
        return None

    def use_collate(self):         # default
        return None

    @staticmethod
    def do_forward(*args, **kwargs):
        raise NotImplementedError


class RankingDataset(Dataset):
    """
    Base template and functionality for pair-wise learning and evaluation.
    """
    def __init__(self, ranking_file):
        # expects to read (user, item1, item2) triplets where item1 > item2 for user
        self.samples: pd.DataFrame = pd.read_hdf(ranking_file + '.h5')
        # use BPR loss for ranking
        self.loss_fn = BPR_loss
        # w hyperparameter for dynamic negative sampling
        self.w = 0.0

    def _negative_sampling_probs(self, negative_ratings: np.ndarray, type='sum_dynamic'):
        if type == 'sum':
            # simple way to boost hard negatives i.e. negatives with bigger ratings
            probs = negative_ratings / sum(negative_ratings)  # give more chances to hard negatives
        elif type == 'sum_dynamic':
            # modified version of sum that gives more or less boost to higher rated items based on w
            negative_ratings_squared = negative_ratings**self.w
            probs = negative_ratings_squared / sum(negative_ratings_squared)  # give more chances to hard negatives
        elif type == 'softmax':
            # balanced but too expensive (would not recommend)
            probs = softmax(torch.FloatTensor(negative_ratings), dim=0).numpy()
        else:
            probs = None    # uniform
        return probs

    def __getitem__(self, item):
        # return (user ID, item1 ID, item2 ID) triplets
        data = self.samples.iloc[item]
        # sample negative from options. Note: This selects just one out of potentially a lot of choices e.g 50 - 300
        probs = self._negative_sampling_probs(np.array(data['negative_ratings']))
        negative = np.random.choice(data['negative_movieIds'], p=probs)
        return data['userId'], data['positive_movieId'], negative

    def __len__(self):
        return len(self.samples)

    def calculate_loss(self, out_pos, out_neg):
        return self.loss_fn(out_pos, out_neg)

    def get_graph(self, device):   # default
        return None

    def use_collate(self):         # default
        return None

    @staticmethod
    def do_forward(*args, **kwargs):
        raise NotImplementedError


def BPR_loss(out_pos, out_neg):
    return torch.sum(-torch.log(torch.sigmoid(out_pos - out_neg)))
