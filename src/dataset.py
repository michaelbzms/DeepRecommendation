import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from globals import item_metadata_file, user_ratings_file, user_embeddings_file


def create_user_embedding(user_ratings: pd.DataFrame, metadata: pd.DataFrame):
    avg_rating = user_ratings['rating'].mean()
    return ((user_ratings['rating'] - avg_rating) * metadata.loc[user_ratings['movieId']]['features'].values).mean()    # TODO: sum or mean?


class MovieLensDataset(Dataset):
    # Static: same for all instances -> shared across train-val-test sets
    print('Initializing common dataset prerequisites...')
    metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
    user_embeddings: pd.DataFrame = pd.read_hdf(user_embeddings_file + '.h5')
    print('Done')

    def __init__(self, file):
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')

    def __getitem__(self, item):
        """ returns (item input, user input, target) """
        data = self.samples.iloc[item]
        rating = float(data['rating'])
        item_tensor = torch.FloatTensor(self.metadata.loc[data['movieId']]['features'])
        user_tensor = torch.FloatTensor(self.user_embeddings.loc[data['userId']][0])
        return item_tensor, user_tensor, rating

    def create_user_embedding(self, user_ratings: pd.DataFrame):
        # Note: Old way. This is for just-in-time user embedding creation. Better to do pre-do this for the whole class.
        avg_rating = user_ratings['rating'].mean()
        return torch.FloatTensor(((user_ratings['rating'] - avg_rating) * self.metadata.loc[user_ratings['movieId']]['features'].values).mean())   # TODO: sum or mean?

    def __len__(self):
        return self.samples.shape[0]

    @staticmethod
    def get_metadata_dim():
        return len(MovieLensDataset.metadata['features'].iloc[0])
