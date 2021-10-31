import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from globals import item_metadata_file, user_ratings_file, user_embeddings_file, audio_features_file


def create_user_embedding(user_ratings: pd.DataFrame, metadata: pd.DataFrame):
    avg_rating = user_ratings['rating'].mean()
    return ((user_ratings['rating'] - avg_rating) * metadata.loc[user_ratings['movieId']]['features'].values).mean()    # TODO: sum or mean?


class MovieLensDatasetPreloaded(Dataset):
    # Static: same for all instances -> shared across train-val-test sets
    print('Initializing common dataset prerequisites (preloaded user embeddings)...')
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
        return len(MovieLensDatasetPreloaded.metadata['features'].iloc[0])


class MovieLensDataset(Dataset):
    print('Initializing common dataset prerequisites...')
    metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
    audio: pd.DataFrame = pd.read_csv(audio_features_file + '.csv', sep=';', index_col='movieId')
    audio.drop(['primaryTitle', 'fileName'], axis=1, inplace=True)   # drop non-features if they exist
    user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')
    print('Done')

    def __init__(self, file):
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')

    def __getitem__(self, item):
        """ returns ((item metadata, item audio), [(ratings, items metadata, items audio)], target rating) """
        data = self.samples.iloc[item]
        target_rating = float(data['rating'])

        item_metadata = torch.FloatTensor(self.metadata.loc[data['movieId']]['features'])
        item_audio = torch.FloatTensor(self.audio.loc[data['movieId']].astype(np.float64))

        user_data = self.user_ratings.loc[data['userId']]
        user_ratings = torch.Tensor(user_data['rating'] - user_data['meanRating'])

        user_items_metadata = torch.FloatTensor(np.stack(self.metadata.loc[user_data['movieId']]['features'].values))
        # TODO: audio features have weird values like 9.34E+10. Should be clipped beforehand to [0, 1]?
        user_items_audio = torch.FloatTensor(self.audio.loc[user_data['movieId']].values)

        return (item_metadata, item_audio), (user_ratings, user_items_metadata, user_items_audio), target_rating

    def __len__(self):
        return self.samples.shape[0]

    @staticmethod
    def get_metadata_dim():
        return len(MovieLensDataset.metadata['features'].iloc[0]), MovieLensDataset.audio.shape[1]
