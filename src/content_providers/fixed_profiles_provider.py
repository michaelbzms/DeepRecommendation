import numpy as np
import pandas as pd

from neural_collaborative_filtering.content_providers import ContentProvider
from globals import item_metadata_file, user_embeddings_file, full_matrix_file, user_embeddings_with_val_file
from util import one_hot_encode


""" Future work (TODO):
- Perform target interaction masking in Basic NCF too by precalculating only the sum of the user profile
and then during inference subtract the candidate profile to undo its addition and then divide by (N-1).
"""


class FixedProfilesProvider(ContentProvider):
    # noinspection PyTypeChecker
    def __init__(self, include_val_ratings_to_user_profiles=False):
        self.metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
        emb_file = user_embeddings_with_val_file if include_val_ratings_to_user_profiles else user_embeddings_file
        self.user_embeddings: pd.DataFrame = pd.read_hdf(emb_file + '.h5')

    def get_item_profile(self, itemID):
        return self.metadata.loc[itemID, :].values

    def get_user_profile(self, userID):
        return self.user_embeddings.loc[userID, :].values

    def get_num_items(self):
        return self.metadata.shape[0]

    def get_num_users(self):
        return self.user_embeddings.shape[0]

    def get_item_feature_dim(self):
        return self.metadata.shape[1]


class FixedItemProfilesOnlyProvider(ContentProvider):
    # noinspection PyTypeChecker
    def __init__(self):
        self.metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
        # load util matrix with all users and all items
        util_matrix: pd.DataFrame = pd.read_csv(full_matrix_file + '.csv')
        self.all_user_ids = np.array(sorted(util_matrix['userId'].unique()))

    def get_item_profile(self, itemID):
        return self.metadata.loc[itemID, :].values

    def get_user_profile(self, userID):
        return one_hot_encode(userID, self.all_user_ids)

    def get_num_items(self):
        return self.metadata.shape[0]

    def get_num_users(self):
        return len(self.all_user_ids)

    def get_item_feature_dim(self):
        return self.metadata.shape[1]
