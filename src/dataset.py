import torch
from torch.utils.data import Dataset
import pandas as pd

from globals import item_metadata_file, user_ratings_file


class MovieLensDataset(Dataset):
    # Static: same for all instances -> shared across train-val-test sets
    metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
    user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')

    def __init__(self, file):
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')

    def __getitem__(self, item):
        """ returns (item input, user input, target) """
        data = self.samples.iloc[item]
        rating = float(data['rating'])
        item_tensor = torch.FloatTensor(self.metadata.loc[data['movieId']]['features'])
        user_ratings: pd.DataFrame = self.user_ratings.loc[data['userId']]
        user_tensor = self.create_user_embedding(user_ratings)
        return item_tensor, user_tensor, rating

    def create_user_embedding(self, user_ratings: pd.DataFrame):
        avg_rating = user_ratings['rating'].mean()
        return torch.FloatTensor(((user_ratings['rating'] - avg_rating) * self.metadata.loc[user_ratings['movieId']]['features'].values).mean())   # TODO: sum or mean?

    def __len__(self):
        return self.samples.shape[0]

    @staticmethod
    def get_metadata_dim():
        return len(MovieLensDataset.metadata['features'].iloc[0])
