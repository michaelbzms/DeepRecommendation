import pandas as pd
import torch
import numpy as np

from globals import user_ratings_file, train_set_file, val_set_file, test_set_file
from gnns.datasets.GNN_dataset import GNN_Dataset


class MovieLensGNNDataset(GNN_Dataset):
    # TODO
    print('Initializing common dataset prerequisites ...')
    user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')
    train_set = pd.read_csv(train_set_file + '.csv')
    val_set = pd.read_csv(val_set_file + '.csv')
    test_set = pd.read_csv(test_set_file + '.csv')
    print('Calculating all unique users and items')
    all_users = np.array(sorted(list(set(train_set['userId']).union(set(val_set['userId'])).union(set(test_set['userId'])))))
    all_items = np.array(sorted(list(set(train_set['movieId']).union(set(val_set['movieId'])).union(set(test_set['movieId'])))))
    print('Done')

