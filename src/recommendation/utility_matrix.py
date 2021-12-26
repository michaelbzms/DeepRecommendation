import numpy as np
import pandas as pd

from globals import train_set_file, test_set_file


class UtilityMatrix:
    def __init__(self, file):
        """ Read utility matrix in sparse (userId, movieId, rating) format """
        self.file = file
        self.sparse_matrix: pd.DataFrame = pd.read_csv(file + '.csv')
        # go from sparse to dense format:
        self.matrix = self.sparse_matrix.pivot(index='userId', columns='movieId', values='rating')

        self.item_avg_ratings = self.matrix.mean(axis=0)    # NaNs ignored
        self.user_avg_ratings = self.matrix.mean(axis=1)    # NaNs ignored

    def get_file(self):
        return self.file

    def get_items_mean_ratings(self):
        return self.item_avg_ratings.copy()

    def get_users_mean_ratings(self):
        return self.user_avg_ratings.copy()

    def get_user_mean_rating(self, userId: int):
        return self.user_avg_ratings.loc[userId]

    def get_item_mean_rating(self, itemId: str):
        return self.item_avg_ratings.loc[itemId]

    def get_overall_mean_rating(self):
        return self.matrix.sum().sum() / self.matrix.count().sum()

    def get_all_users(self):
        return np.sort(self.sparse_matrix['userId'].unique())

    def get_all_items(self):
        return np.sort(self.sparse_matrix['movieId'].unique())


if __name__ == '__main__':
    """ To test it """
    u = UtilityMatrix('../' + train_set_file)
