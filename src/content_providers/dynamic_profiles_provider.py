import torch
import pandas as pd
import numpy as np

from neural_collaborative_filtering.content_providers import DynamicContentProvider
from util import multi_hot_encode
from globals import item_metadata_file, user_ratings_file


class DynamicProfilesProvider(DynamicContentProvider):
    # noinspection PyTypeChecker
    def __init__(self):
        self.metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
        self.user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')

    def get_item_profile(self, itemID):  # ID or IDs
        return self.metadata.loc[itemID, :].values

    def get_num_items(self):                     # TODO: what if we store more than are in our samples?
        return self.metadata.shape[0]

    def get_num_users(self):                     # TODO: what if we store more than are in our samples?
        return self.user_ratings.shape[0]

    def get_item_feature_dim(self):
        return self.metadata.shape[1]

    def collate_interacted_items(self, batch, for_ranking: bool, ignore_ratings=False):
        """
        Combine __getitem__() with this custom collate_fn to return for each batch in a data loader:
          > candidate_items_batch: (B, F)  B items with their features
          > rated_items_features: (I, F) I rated items with their features
          > user_matrix: (B, I) a subarray (not exactly) of the utility matrix with the (normalized) ratings of B users on I items.
        The order must match rated_items_feature's order on I axis.

        It is more efficient to do all these batch-wise. One would have to repeat the same process
        for other contents as well.
        """

        # turn per-row to per-column
        batch_data = list(zip(*batch))

        # get item profiles and stack them
        candidate_items = torch.FloatTensor(self.get_item_profile(itemID=batch_data[1]))
        if for_ranking:
            targets_or_items2 = torch.FloatTensor(self.get_item_profile(itemID=batch_data[2]))
        else:
            targets_or_items2 = torch.FloatTensor(batch_data[2])

        # for the user part we do all the work here.
        # get user ids in batch and their ratings
        user_ids = list(batch_data[0])
        user_ratings = self.user_ratings.loc[user_ids]

        # get unique item ids in batch
        rated_items_ids = np.sort(np.unique(np.concatenate(user_ratings['movieId'].values)))

        # multi-hot encode sparse ratings into a matrix form
        user_matrix = multi_hot_encode(user_ratings['movieId'], rated_items_ids).astype(np.float64)
        if not ignore_ratings:
            # Note: This will work but ONLY IF ratings are ORDERED by movieId when we create the dataset.
            # Else the ratings will be misplaced! Be careful!
            user_matrix[user_matrix == 1] = np.concatenate((user_ratings['rating'] - ((user_ratings['meanRating'] + 2.5) / 2)).values)
            # check: e.g. user_matrix[0, rated_movies == 'tt0114709']
        user_matrix = torch.FloatTensor(user_matrix)  # convert to tensor

        # get features for all rated items in batch
        rated_items = torch.FloatTensor(self.get_item_profile(rated_items_ids))

        return candidate_items, rated_items, user_matrix, targets_or_items2
