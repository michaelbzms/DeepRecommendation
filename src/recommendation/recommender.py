from abc import ABC
import numpy as np
import pandas as pd

from recommendation.utility_matrix import UtilityMatrix


class Recommender(ABC):
    def __init__(self, name):
        self.name = name

    def fit(self, train_utility_matrix: UtilityMatrix, filepath):
        pass

    def calculate_MSE(self, test_utility_matrix: UtilityMatrix, filepath):
        pass


class NaiveBaselineRecommender(Recommender):
    def __init__(self):
        super(NaiveBaselineRecommender, self).__init__("NaiveBaselineRecommender")
        self.predict_rating = None

    def fit(self, train_utility_matrix: UtilityMatrix, filepath=None):
        self.predict_rating = train_utility_matrix.get_overall_mean_rating()

    def calculate_MSE(self, test_utility_matrix: UtilityMatrix, filepath=None):
        SSE = np.power(test_utility_matrix.sparse_matrix['rating'].values - self.predict_rating, 2)
        return np.mean(SSE)


class PerItemBaselineRecommender(Recommender):
    def __init__(self):
        super(PerItemBaselineRecommender, self).__init__("PerItemBaselineRecommender")
        self.per_item_predictions = None

    def fit(self, train_utility_matrix: UtilityMatrix, filepath=None):
        self.per_item_predictions = train_utility_matrix.get_items_mean_ratings()

    def calculate_MSE(self, test_utility_matrix: UtilityMatrix, filepath=None):
        # movie_ratings = test_utility_matrix.sparse_matrix.pivot_table(index=['movieId'], values='rating', aggfunc=list)
        # movie_ratings.apply(lambda x: np.array(x))
        SSE = 0.0
        count = 0
        for _, row in test_utility_matrix.sparse_matrix.iterrows():
            SSE += (row['rating'] - self.per_item_predictions.loc[row['movieId']])**2
            count += 1
        return SSE / count
