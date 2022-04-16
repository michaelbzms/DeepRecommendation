import pandas as pd

from neural_collaborative_filtering.content_providers import ContentProvider
from globals import item_metadata_file, user_embeddings_file


class FixedProfilesProvider(ContentProvider):
    # noinspection PyTypeChecker
    def __init__(self):
        self.metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
        self.user_embeddings: pd.DataFrame = pd.read_hdf(user_embeddings_file + '.h5')

    def get_item_profile(self, itemID):
        return self.metadata.loc[itemID, :].values

    def get_user_profile(self, userID):
        return self.user_embeddings.loc[userID, :].values

    def __create_user_embedding(self, user_ratings: pd.DataFrame):  # not used currently TODO
        # Note: Old way for non fixed embeddings.
        # This is for just-in-time user embedding creation. Better to do pre-do this for the whole class.
        avg_rating = user_ratings['rating'].mean()
        return ((user_ratings['rating'] - avg_rating) * self.metadata.loc[user_ratings['movieId']].values).mean(axis=0)   # TODO: might be wrong

    def get_num_items(self):                     # TODO: what if we store more than are in our samples?
        return self.metadata.shape[0]

    def get_num_users(self):                     # TODO: what if we store more than are in our samples?
        return self.user_embeddings.shape[0]

    def get_item_feature_dim(self):
        return self.metadata.shape[1]
