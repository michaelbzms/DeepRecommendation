from globals import train_set_file, test_set_file
from recommendation.recommender import NaiveBaselineRecommender, PerItemBaselineRecommender
from recommendation.utility_matrix import UtilityMatrix


if __name__ == '__main__':
    train_matrix = UtilityMatrix(train_set_file)
    test_matrix = UtilityMatrix(test_set_file)

    recommender_list = [NaiveBaselineRecommender(), PerItemBaselineRecommender()]

    for rec in recommender_list:
        rec.fit(train_matrix)
        mse = rec.calculate_MSE(test_matrix)
        print('Recommender:', rec.name, '- MSE:', mse)
