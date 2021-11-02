import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from globals import item_metadata_file, user_ratings_file, user_embeddings_file, audio_features_file, features_to_use


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


def multihot_encode(actual_values, ordered_possible_values) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform(actual_values)
    return binary_format


def my_collate_fn(batch):
    # turn per-row to per-column
    batch_data = list(zip(*batch))
    # stack torch tensors from dataset
    candidate_items = torch.stack(batch_data[0])
    targets = torch.FloatTensor(batch_data[2])
    # for the user part we do all the work here.
    # get user ids in batch and their ratings
    user_ids = list(batch_data[1])
    user_ratings = MovieLensDataset.user_ratings.loc[user_ids]
    # get unique item ids in batch
    rated_items_ids = np.unique(np.concatenate(user_ratings['movieId'].values))   # TODO: must return ordered list (seems to)
    # multi-hot encode sparse ratings into a matrix form
    user_matrix = multihot_encode(user_ratings['movieId'], rated_items_ids).astype(np.float64)
    # TODO: This will work but ONLY IF ratings are ORDERED by movieId when we create the dataset. Else the ratings will be misplaced! Be careful!
    user_matrix[user_matrix == 1] = np.concatenate((user_ratings['rating'] - user_ratings['meanRating']).values)
    # check: e.g. user_matrix[0, rated_movies == 'tt0114709']
    user_matrix = torch.FloatTensor(user_matrix)       # convert to tensor
    # get features for all rated items in batch
    if features_to_use == 'metadata':
        rated_items = torch.FloatTensor(MovieLensDataset.metadata.loc[rated_items_ids]['features'].values)
    elif features_to_use == 'audio':
        rated_items = torch.FloatTensor(MovieLensDataset.audio.loc[rated_items_ids].astype(np.float64).values)
    elif features_to_use == 'all' or features_to_use == 'both':
        rated_items = torch.cat((torch.FloatTensor(MovieLensDataset.metadata.loc[rated_items_ids]['features'].values),
                                 torch.FloatTensor(MovieLensDataset.audio.loc[rated_items_ids].astype(np.float64).values)), dim=1)
    else:
        raise Exception('Invalid features_to_use parameter in dataset')

    return candidate_items, rated_items, user_matrix, targets


class MovieLensDataset(Dataset):
    print('Initializing common dataset prerequisites...')
    metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
    audio: pd.DataFrame = pd.read_csv(audio_features_file + '.csv', sep=';', index_col='movieId')
    audio.drop(['primaryTitle', 'fileName'], axis=1, inplace=True)   # drop non-features if they exist
    audio = audio.astype(np.float64)
    user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')
    print('Done')

    def __init__(self, file):      # 'metadata', 'audio' or 'all'
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')

    def __getitem__(self, item):
        """ returns ((item metadata, item audio), [(ratings, items metadata, items audio)], target rating) """
        # get sample
        data = self.samples.iloc[item]
        # get target rating
        target_rating = float(data['rating'])
        # get candidate item features
        if features_to_use == 'metadata':
            candidate_items = torch.FloatTensor(self.metadata.loc[data['movieId']]['features'])
        elif features_to_use == 'audio':
            candidate_items = torch.FloatTensor(self.audio.loc[data['movieId']].astype(np.float64))
        elif features_to_use == 'all' or features_to_use == 'both':
            candidate_items = torch.cat((torch.FloatTensor(self.metadata.loc[data['movieId']]['features']),
                                         torch.FloatTensor(self.audio.loc[data['movieId']].astype(np.float64))), dim=1)
        else:
            raise Exception('Invalid features_to_use parameter in dataset')
        # for the user part only forward the user's ID. We will mass collect info in batch-level
        return candidate_items, data['userId'], target_rating

    def __len__(self):
        return self.samples.shape[0]

    @staticmethod
    def get_metadata_dim():
        if features_to_use == 'metadata':
            return len(MovieLensDataset.metadata['features'].iloc[0])
        elif features_to_use == 'audio':
            return MovieLensDataset.audio.shape[1]
        elif features_to_use == 'all' or features_to_use == 'both':
            return len(MovieLensDataset.metadata['features'].iloc[0]) + MovieLensDataset.audio.shape[1]
        else:
            raise Exception('Invalid features_to_use parameter in dataset')
