import torch
import pandas as pd
import numpy as np

from globals import item_metadata_file, user_ratings_file, audio_features_file, features_to_use
from neural_collaborative_filtering.datasets.dynamic_dataset import DynamicDataset
from util import multi_hot_encode


# TODO: Problem -> this dataset can not be expressed as a simple Content provider because user profiles are created at forward()


class DynamicMovieLensDataset(DynamicDataset):
    print('Initializing common dataset prerequisites...')
    metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
    if features_to_use == 'audio' or features_to_use == 'all':
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

    @staticmethod
    def get_number_of_items():
        return DynamicMovieLensDataset.metadata.shape[0]

    @staticmethod
    def get_all_itemIds_sorted():
        return DynamicMovieLensDataset.item_ids

    @staticmethod
    def get_sorted_item_names():
        return DynamicMovieLensDataset.item_names.loc[DynamicMovieLensDataset.item_ids].values.flatten()

    @staticmethod
    def get_item_feature_dim():
        if features_to_use == 'metadata':
            return len(DynamicMovieLensDataset.metadata.iloc[0])
        elif features_to_use == 'audio':
            return DynamicMovieLensDataset.audio.shape[1]
        elif features_to_use == 'all' or features_to_use == 'both':
            return len(DynamicMovieLensDataset.metadata.iloc[0]) + DynamicMovieLensDataset.audio.shape[1]
        else:
            raise Exception('Invalid features_to_use parameter in dataset')

    @staticmethod
    def use_collate():
        return my_collate_fn2


class NamedDynamicMovieLensDataset(DynamicMovieLensDataset):
    def __init__(self, file):
        super(NamedDynamicMovieLensDataset, self).__init__(file)

    def __getitem__(self, item):
        x = super(NamedDynamicMovieLensDataset, self).__getitem__(item)
        # get sample
        data = self.samples.iloc[item]
        name = DynamicMovieLensDataset.item_names.loc[data['movieId']].values
        return x + (name, )      # tuple concat


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
    user_ratings = DynamicMovieLensDataset.user_ratings.loc[user_ids]
    # get unique item ids in batch
    rated_items_ids = np.unique(np.concatenate(user_ratings['movieId'].values))   # TODO: must return ordered list (seems to)
    # multi-hot encode sparse ratings into a matrix form
    user_matrix = multi_hot_encode(user_ratings['movieId'], rated_items_ids).astype(np.float64)
    if not ignore_ratings:
        # TODO: This will work but ONLY IF ratings are ORDERED by movieId when we create the dataset. Else the ratings will be misplaced! Be careful!
        user_matrix[user_matrix == 1] = np.concatenate((user_ratings['rating'] - user_ratings['meanRating']).values)
        # check: e.g. user_matrix[0, rated_movies == 'tt0114709']
    user_matrix = torch.FloatTensor(user_matrix)       # convert to tensor
    # get features for all rated items in batch
    if features_to_use == 'metadata':
        rated_items = torch.FloatTensor(np.stack(DynamicMovieLensDataset.metadata.loc[rated_items_ids].values))
    elif features_to_use == 'audio':
        rated_items = torch.FloatTensor(DynamicMovieLensDataset.audio.loc[rated_items_ids].astype(np.float64).values)
    elif features_to_use == 'all' or features_to_use == 'both':
        rated_items = torch.cat((torch.FloatTensor(np.stack(DynamicMovieLensDataset.metadata.loc[rated_items_ids].values)),
                                 torch.FloatTensor(DynamicMovieLensDataset.audio.loc[rated_items_ids].astype(np.float64).values)), dim=1)
    else:
        raise Exception('Invalid features_to_use parameter in dataset')

    if not with_names:
        return candidate_items, rated_items, user_matrix, targets
    else:
        candidate_names = np.vstack(batch_data[-1]).flatten() if len(batch_data[-1]) > 1 else batch_data[-1]
        rated_item_names = DynamicMovieLensDataset.item_names.loc[rated_items_ids]
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
    user_ratings = DynamicMovieLensDataset.user_ratings.loc[user_ids]
    # multi-hot encode sparse ratings into a matrix form
    all_item_ids = DynamicMovieLensDataset.get_all_itemIds_sorted()
    user_matrix = multi_hot_encode(user_ratings['movieId'], all_item_ids.tolist()).astype(np.float64)
    if not ignore_ratings:
        # TODO: This will work but ONLY IF ratings are ORDERED by movieId when we create the dataset. Else the ratings will be misplaced! Be careful!
        user_matrix[user_matrix == 1] = np.concatenate((user_ratings['rating'] - user_ratings['meanRating']).values)
        # check: e.g. user_matrix[0, rated_movies == 'tt0114709']
    user_matrix = torch.FloatTensor(user_matrix)  # convert to tensor
    # get features for ALL items
    if features_to_use == 'metadata':
        all_item_features = torch.FloatTensor(np.stack(DynamicMovieLensDataset.metadata.loc[all_item_ids].values))    # (!) Note: .loc important to reorder all item features
    elif features_to_use == 'audio':
        all_item_features = torch.FloatTensor(DynamicMovieLensDataset.audio.loc[all_item_ids].astype(np.float64).values)
    elif features_to_use == 'all' or features_to_use == 'both':
        all_item_features = torch.cat((torch.FloatTensor(np.stack(DynamicMovieLensDataset.metadata.loc[all_item_ids].values)),
                                       torch.FloatTensor(DynamicMovieLensDataset.audio.loc[all_item_ids].astype(np.float64).values)), dim=1)
        # TODO: audio features have more imbdIds?
    else:
        raise Exception('Invalid features_to_use parameter in dataset')

    if not with_names:
        return candidate_items, all_item_features, user_matrix, targets
    else:
        candidate_names = np.vstack(batch_data[-1]).flatten() if len(batch_data[-1]) > 1 else batch_data[-1]
        all_item_names = DynamicMovieLensDataset.item_names.loc[all_item_ids]
        return candidate_items, all_item_features, user_matrix, targets, candidate_names, all_item_names
