from globals import train_set_file, test_set_file
from models.AdvancedNCF import AdvancedNCF
from models.AttentionNCF import AttentionNCF
from models.NCF import load_model
from recommendation.recommender import NaiveBaselineRecommender, PerItemBaselineRecommender, NCF_Recommender
from recommendation.utility_matrix import UtilityMatrix


if __name__ == '__main__':
    train_matrix = UtilityMatrix(train_set_file)
    test_matrix = UtilityMatrix(test_set_file)

    recommender_list = [
        NaiveBaselineRecommender(),
        PerItemBaselineRecommender(),
        NCF_Recommender(load_model('../models/AdvancedNCF_audio_meta.pt', AdvancedNCF),
                        use_metadata=True, use_audio=True),
        NCF_Recommender(load_model('../models/AttentionNCF_audio_meta.pt', AttentionNCF),
                        use_metadata=True, use_audio=True)
    ]

    for rec in recommender_list:
        # fit on train matrix (if NN then it won't do anything, loads already fitted model)
        rec.fit(train_matrix)
        # evaluate on test matrix
        mse = rec.calculate_MSE(test_matrix)
        rmse = rec.calculate_RMSE(test_matrix)
        # print results
        print('Recommender:', rec.name, f'- MSE: {mse:.3f} - RMSE: {rmse:.3f}')
