import torch
import pandas as pd
import numpy as np

from globals import user_ratings_file, train_set_file, val_set_file, test_set_file
from neural_collaborative_filtering.datasets.fixed_dataset import FixedDataset
from util import one_hot_encode


# TODO: use_utility_matrix_instead = False


class OneHotMovieLensDataset(FixedDataset):
    print('Initializing common dataset prerequisites ...')
    user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')
    train_set = pd.read_csv(train_set_file + '.csv')
    val_set = pd.read_csv(val_set_file + '.csv')
    test_set = pd.read_csv(test_set_file + '.csv')
    print('Calculating all unique users and items')
    all_users = np.array(sorted(list(set(train_set['userId']).union(set(val_set['userId'])).union(set(test_set['userId'])))))
    all_items = np.array(sorted(list(set(train_set['movieId']).union(set(val_set['movieId'])).union(set(test_set['movieId'])))))
    print('Done')

    def __init__(self, file: str):
        if file == train_set_file:
            self.set = OneHotMovieLensDataset.train_set
        elif file == val_set_file:
            self.set = OneHotMovieLensDataset.val_set
        elif file == test_set_file:
            self.set = OneHotMovieLensDataset.test_set
        else:
            raise Exception('Invalid filepath for OneHot dataset')

    def __getitem__(self, item):
        data = self.set.iloc[item]
        item_vec = one_hot_encode(data['movieId'], OneHotMovieLensDataset.all_items)
        user_vec = one_hot_encode(data['userId'], OneHotMovieLensDataset.all_users)
        return torch.tensor(item_vec), torch.tensor(user_vec), float(data['rating'])

    def __len__(self):
        return self.set.shape[0]

    @staticmethod
    def get_number_of_users():
        return len(OneHotMovieLensDataset.all_users)

    @staticmethod
    def get_number_of_items():
        return len(OneHotMovieLensDataset.all_items)

    @staticmethod
    def get_item_feature_dim():  # aka F
        return len(OneHotMovieLensDataset.all_items)    # because of one-hot-encoding
