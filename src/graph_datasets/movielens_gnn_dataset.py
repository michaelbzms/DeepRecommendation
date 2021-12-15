import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import networkx as nx
from torch_geometric.utils import from_networkx

from globals import user_ratings_file, train_set_file, val_set_file, test_set_file
from gnns.datasets.GNN_dataset import GNN_Dataset


def create_item_features():
    return create_onehot_features(MovieLensGNNDataset.all_items)


def create_user_features():
    return create_onehot_features(MovieLensGNNDataset.all_users)


def create_onehot_features(all):
    return torch.eye(len(all), dtype=torch.float)


class MovieLensGNNDataset(GNN_Dataset):
    # TODO
    print('Initializing common dataset prerequisites ...')
    user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')
    train_set = pd.read_csv(train_set_file + '.csv')
    val_set = pd.read_csv(val_set_file + '.csv')
    test_set = pd.read_csv(test_set_file + '.csv')

    print('Calculating all unique users and items')
    all_users = np.array(sorted(list(set(train_set['userId']).union(set(val_set['userId'])).union(set(test_set['userId'])))))
    all_items = np.array(sorted(list(set(train_set['movieId']).union(set(val_set['movieId'])).union(set(test_set['movieId'])))))
    print('Done')

    def __init__(self, file: str):
        if file == train_set_file:
            self.set = MovieLensGNNDataset.train_set
            self.graph_edges = MovieLensGNNDataset.train_set     # but dynamically reduce them
            self.reduce_per_batch = True
        elif file == val_set_file:
            self.set = MovieLensGNNDataset.val_set
            self.graph_edges = MovieLensGNNDataset.train_set
            self.reduce_per_batch = False
        elif file == test_set_file:
            self.set = MovieLensGNNDataset.test_set
            self.graph_edges = MovieLensGNNDataset.train_set.append(MovieLensGNNDataset.val_set)  # use validation edges too
            self.reduce_per_batch = False
        else:
            raise Exception('Invalid filepath for OneHot dataset')

        print('Creating graph...')
        all_users = MovieLensGNNDataset.all_users
        all_items = MovieLensGNNDataset.all_items

        all_users_index = {u: ind for ind, u in enumerate(all_users)}
        all_items_index = {i: ind for ind, i in enumerate(all_items)}

        edge_index = [[all_users_index[u] for u in self.graph_edges['userId']],
                      [all_items_index[i] for i in self.graph_edges['movieId']]]
        rev_edge_index = [edge_index[1], edge_index[0]]

        # TODO: use rating - avg user rating instead, should be better
        edge_attr = [[rating] for rating in self.graph_edges['rating']]

        # TODO: If I want to remove edges that are targets in the batch maybe I need to delay graph creation until the batch?
        self.train_graph = HeteroData(
            user={'x': create_user_features()},  # NUM_USERS x FEAT_USERS
            item={'x': create_item_features()},  # NUM_ITEMS x FEAT_ITEMS
            user__rates__item={'edge_index': torch.tensor(edge_index, dtype=torch.long),   # 2 x NUM_EDGES
                               'edge_attr': torch.tensor(edge_attr, dtype=torch.float)},
            items__ratedby__user={'edge_index': torch.tensor(rev_edge_index, dtype=torch.long),
                                  'edge_attr': torch.tensor(edge_attr, dtype=torch.float)}
        )
        print(self.train_graph)

        # TODO: OMG There is a from networkx method! But it won't make a hetero graph
        # TODO: use rating - avg user rating instead, should be better
        # g = nx.Graph([(data['userId'], data['movieId'], {'weight': data['rating']}) for _, data in self.graph_edges.iterrows()])
        # self.train_graph2 = from_networkx(g)
        # print(self.train_graph2)
        # pass
