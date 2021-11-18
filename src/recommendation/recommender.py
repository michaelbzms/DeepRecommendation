from abc import ABC

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dynamic_dataset import NamedMovieLensDataset, my_collate_fn
from globals import val_batch_size
from models import NCF
from recommendation.utility_matrix import UtilityMatrix


class Recommender(ABC):
    def __init__(self, name):
        self.name = name

    def fit(self, train_utility_matrix: UtilityMatrix):
        pass    # do nothing by default

    def calculate_MSE(self, test_utility_matrix: UtilityMatrix):
        raise Exception('Not Implemented')

    def calculate_RMSE(self, test_utility_matrix: UtilityMatrix):
        return np.sqrt(self.calculate_MSE(test_utility_matrix))


class NaiveBaselineRecommender(Recommender):
    def __init__(self):
        super(NaiveBaselineRecommender, self).__init__("NaiveBaselineRecommender")
        self.predict_rating = None

    def fit(self, train_utility_matrix: UtilityMatrix):
        self.predict_rating = train_utility_matrix.get_overall_mean_rating()

    def calculate_MSE(self, test_utility_matrix: UtilityMatrix):
        SSE = np.power(test_utility_matrix.sparse_matrix['rating'].values - self.predict_rating, 2)
        return np.mean(SSE)


class PerItemBaselineRecommender(Recommender):
    def __init__(self):
        super(PerItemBaselineRecommender, self).__init__("PerItemBaselineRecommender")
        self.per_item_predictions = None

    def fit(self, train_utility_matrix: UtilityMatrix):
        self.per_item_predictions = train_utility_matrix.get_items_mean_ratings()

    def calculate_MSE(self, test_utility_matrix: UtilityMatrix):
        # movie_ratings = test_utility_matrix.sparse_matrix.pivot_table(index=['movieId'], values='rating', aggfunc=list)
        # movie_ratings.apply(lambda x: np.array(x))
        SSE = 0.0
        count = 0
        for _, row in test_utility_matrix.sparse_matrix.iterrows():
            SSE += (row['rating'] - self.per_item_predictions.loc[row['movieId']])**2
            count += 1
        return SSE / count


class NCF_Recommender(Recommender):
    def __init__(self, model: NCF, use_metadata, use_audio):
        super(NCF_Recommender, self).__init__(type(model).__name__ +
                                              ('_metadata' if use_metadata else '') +
                                              ('_audio' if use_audio else ''))
        self.model = model
        self.use_metadata = use_metadata
        self.use_audio = use_audio
        self.MSE = None

    def calculate_MSE(self, test_utility_matrix):
        if self.MSE is None:      # if not calculated already
            # load dataset and test loader for it
            test_set_file = test_utility_matrix.get_file()
            test_dataset = NamedMovieLensDataset(test_set_file)
            test_loader = DataLoader(test_dataset, batch_size=val_batch_size, collate_fn=my_collate_fn)

            criterion = nn.MSELoss(reduction='sum')  # don't average the loss as we shall do that ourselves for the whole epoch

            # Calculate val_loss and see if we need to stop
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()  # gradients "off"
            test_sum_loss = 0.0
            test_size = 0
            fitted_values = []
            ground_truth = []
            with torch.no_grad():
                for data in test_loader:
                    # get the input matrices and the target
                    candidate_items, rated_items, user_matrix, y_batch = data
                    # forward model
                    out = self.model(candidate_items.float().to(device), rated_items.float().to(device), user_matrix.float().to(device))
                    # calculate loss
                    loss = criterion(out, y_batch.view(-1, 1).float().to(device))
                    # accumulate validation loss
                    test_sum_loss += loss.detach().item()
                    test_size += len(y_batch)
                    # keep track of fitted values and their actual targets
                    fitted_values.append(out.cpu().detach().numpy())
                    ground_truth.append(y_batch.view(-1, 1).float().cpu().detach().numpy())
            self.MSE = test_sum_loss / test_size
        return self.MSE
