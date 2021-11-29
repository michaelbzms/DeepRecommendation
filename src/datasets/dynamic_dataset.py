import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from globals import item_metadata_file, user_ratings_file, audio_features_file, features_to_use


class MovieLensDataset(Dataset):
    print('Initializing common dataset prerequisites...')
    metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
    audio: pd.DataFrame = pd.read_csv(audio_features_file + '.csv', sep=';', index_col='movieId')
    item_names = pd.DataFrame(index=audio.index, data=audio['primaryTitle'].copy()).loc[metadata.index]
    audio.drop(['primaryTitle', 'fileName'], axis=1, inplace=True)   # drop non-features if they exist
    audio = audio.astype(np.float64)
    user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')
    item_ids = np.array(sorted(metadata.index.to_list()))
    print('Done')

    def __init__(self, file):
        self.samples: pd.DataFrame = pd.read_csv(file + '.csv')

    def __getitem__(self, item):
        # get sample
        data = self.samples.iloc[item]
        # get target rating
        target_rating = float(data['rating'])
        # get candidate item features
        if features_to_use == 'metadata':
            candidate_items = torch.FloatTensor(self.metadata.loc[data['movieId']])
        elif features_to_use == 'audio':
            candidate_items = torch.FloatTensor(self.audio.loc[data['movieId']].astype(np.float64))
        elif features_to_use == 'all' or features_to_use == 'both':
            candidate_items = torch.cat((torch.FloatTensor(self.metadata.loc[data['movieId']]),
                                         torch.FloatTensor(self.audio.loc[data['movieId']].astype(np.float64))))
        else:
            raise Exception('Invalid features_to_use parameter in dataset')
        # for the user part only forward the user's ID. We will mass collect info in batch-level
        return candidate_items, data['userId'], target_rating

    def __len__(self):
        return self.samples.shape[0]

    def get_I(self):
        return self.metadata.shape[0]

    @staticmethod
    def get_all_itemIds_sorted():
        return MovieLensDataset.item_ids

    @staticmethod
    def get_sorted_item_names():
        return MovieLensDataset.item_names.loc[MovieLensDataset.item_ids].values.flatten()

    @staticmethod
    def get_item_feature_dim():
        if features_to_use == 'metadata':
            return len(MovieLensDataset.metadata.iloc[0])
        elif features_to_use == 'audio':
            return MovieLensDataset.audio.shape[1]
        elif features_to_use == 'all' or features_to_use == 'both':
            return len(MovieLensDataset.metadata.iloc[0]) + MovieLensDataset.audio.shape[1]
        else:
            raise Exception('Invalid features_to_use parameter in dataset')

    @staticmethod
    def use_collate():
        return my_collate_fn2


class NamedMovieLensDataset(MovieLensDataset):
    def __init__(self, file):
        super(NamedMovieLensDataset, self).__init__(file)

    def __getitem__(self, item):
        x = super(NamedMovieLensDataset, self).__getitem__(item)
        # get sample
        data = self.samples.iloc[item]
        name = MovieLensDataset.item_names.loc[data['movieId']].values
        return x + (name, )      # tuple concat


def multihot_encode(actual_values, ordered_possible_values) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform(actual_values)
    return binary_format


class MyCollator:
    # In order to use args
    def __init__(self, only_rated=True, with_names=False):
        self.only_rated = only_rated
        self.with_names = with_names

    def __call__(self, batch):
        if self.only_rated:
            return my_collate_fn(batch, with_names=self.with_names)
        else:
            return my_collate_fn2(batch, with_names=self.with_names)


def my_collate_fn(batch, with_names=False, ignore_ratings=False):
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
    if not ignore_ratings:
        # TODO: This will work but ONLY IF ratings are ORDERED by movieId when we create the dataset. Else the ratings will be misplaced! Be careful!
        user_matrix[user_matrix == 1] = np.concatenate((user_ratings['rating'] - user_ratings['meanRating']).values)
        # check: e.g. user_matrix[0, rated_movies == 'tt0114709']
    user_matrix = torch.FloatTensor(user_matrix)       # convert to tensor
    # get features for all rated items in batch
    if features_to_use == 'metadata':
        rated_items = torch.FloatTensor(np.stack(MovieLensDataset.metadata.loc[rated_items_ids].values))
    elif features_to_use == 'audio':
        rated_items = torch.FloatTensor(MovieLensDataset.audio.loc[rated_items_ids].astype(np.float64).values)
    elif features_to_use == 'all' or features_to_use == 'both':
        rated_items = torch.cat((torch.FloatTensor(np.stack(MovieLensDataset.metadata.loc[rated_items_ids].values)),
                                 torch.FloatTensor(MovieLensDataset.audio.loc[rated_items_ids].astype(np.float64).values)), dim=1)
    else:
        raise Exception('Invalid features_to_use parameter in dataset')

    if not with_names:
        return candidate_items, rated_items, user_matrix, targets
    else:
        candidate_names = np.vstack(batch_data[-1]).flatten() if len(batch_data[-1]) > 1 else batch_data[-1]
        rated_item_names = MovieLensDataset.item_names.loc[rated_items_ids]
        return candidate_items, rated_items, user_matrix, targets, candidate_names, rated_item_names


def my_collate_fn2(batch, with_names=False, ignore_ratings=False):
    # turn per-row to per-column
    batch_data = list(zip(*batch))
    # stack torch tensors from dataset
    candidate_items = torch.stack(batch_data[0])
    targets = torch.FloatTensor(batch_data[2])
    # for the user part we do all the work here.
    # get user ids in batch and their ratings
    user_ids = list(batch_data[1])
    user_ratings = MovieLensDataset.user_ratings.loc[user_ids]
    # multi-hot encode sparse ratings into a matrix form
    all_item_ids = MovieLensDataset.get_all_itemIds_sorted()
    user_matrix = multihot_encode(user_ratings['movieId'], all_item_ids).astype(np.float64)
    if not ignore_ratings:
        # TODO: This will work but ONLY IF ratings are ORDERED by movieId when we create the dataset. Else the ratings will be misplaced! Be careful!
        user_matrix[user_matrix == 1] = np.concatenate((user_ratings['rating'] - user_ratings['meanRating']).values)
        # check: e.g. user_matrix[0, rated_movies == 'tt0114709']
    user_matrix = torch.FloatTensor(user_matrix)  # convert to tensor
    # get features for ALL items
    if features_to_use == 'metadata':
        all_item_features = torch.FloatTensor(np.stack(MovieLensDataset.metadata.loc[all_item_ids].values))    # (!) Note: .loc important to reorder all item features
    elif features_to_use == 'audio':
        all_item_features = torch.FloatTensor(MovieLensDataset.audio.loc[all_item_ids].astype(np.float64).values)
    elif features_to_use == 'all' or features_to_use == 'both':
        all_item_features = torch.cat((torch.FloatTensor(np.stack(MovieLensDataset.metadata.loc[all_item_ids].values)),
                                       torch.FloatTensor(MovieLensDataset.audio.loc[all_item_ids].astype(np.float64).values)), dim=1)
        # TODO: audio features have more imbdIds?
    else:
        raise Exception('Invalid features_to_use parameter in dataset')

    if not with_names:
        return candidate_items, all_item_features, user_matrix, targets
    else:
        candidate_names = np.vstack(batch_data[-1]).flatten() if len(batch_data[-1]) > 1 else batch_data[-1]
        all_item_names = MovieLensDataset.item_names.loc[all_item_ids]
        return candidate_items, all_item_features, user_matrix, targets, candidate_names, all_item_names
