import pandas as pd
import numpy as np

from neural_collaborative_filtering.content_providers import ContentProvider
from util import one_hot_encode
from globals import full_matrix_file


class OneHotProvider(ContentProvider):
    def __init__(self):
        # load util matrix with all users and all items
        util_matrix: pd.DataFrame = pd.read_csv(full_matrix_file + '.csv')
        # extract all IDs
        self.all_item_ids = np.array(sorted(util_matrix['movieId'].unique()))
        self.all_user_ids = np.array(sorted(util_matrix['userId'].unique()))

    def get_item_profile(self, itemID):
        return one_hot_encode(itemID, self.all_item_ids)

    def get_user_profile(self, userID):
        return one_hot_encode(userID, self.all_user_ids)

    def get_num_items(self):
        return len(self.all_item_ids)

    def get_num_users(self):
        return len(self.all_user_ids)

    def get_item_feature_dim(self):
        return self.get_num_items()
