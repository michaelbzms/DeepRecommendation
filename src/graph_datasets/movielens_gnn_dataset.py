import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData, Data
import networkx as nx
from torch_geometric.utils import from_networkx, coalesce

from globals import user_ratings_file, train_set_file, val_set_file, test_set_file
from gnns.datasets.GNN_dataset import GNN_Dataset


class MovieLensGNNDataset(GNN_Dataset):
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

        # First sorted items then sorted users as nodes with combined one-hot vector representations
        x = torch.eye(len(all_items) + len(all_users))

        self.all_users_index = {u: ind for ind, u in enumerate(all_users)}
        self.all_items_index = {i: ind for ind, i in enumerate(all_items)}

        edge_index = [[self.all_users_index[u] for u in self.graph_edges['userId']],
                      [self.all_items_index[i] for i in self.graph_edges['movieId']]]
        rev_edge_index = [edge_index[1], edge_index[0]]
        # append backward edges
        edge_index[0] += edge_index[1]
        edge_index[1] += edge_index[0][:self.graph_edges.shape[0]]


        # TODO: use rating - avg user rating instead, should be better
        edge_attr = [[rating] for rating in self.graph_edges['rating']] * 2

        # TODO: If I want to remove edges that are targets in the batch maybe I need to delay graph creation until the batch?
        # maybe I can change edge_index and edge_attr directly
        self.known_graph = Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            num_items=len(all_items)
        )
        print(self.known_graph)
        # remove duplicates
        print('Removing duplicate edges...')
        self.known_graph.edge_index, self.known_graph.edge_attr = coalesce(self.known_graph.edge_index, self.known_graph.edge_attr, reduce='mean')
        print(self.known_graph)

        # TODO: OMG There is a from networkx method! But it won't make a hetero graph
        # TODO: use rating - avg user rating instead, should be better
        # g = nx.Graph([(data['userId'], data['movieId'], {'weight': data['rating']}) for _, data in self.graph_edges.iterrows()])
        # self.train_graph2 = from_networkx(g)
        # print(self.train_graph2)
        # pass

    def __getitem__(self, item):
        """ returns (user_index, item_index, target rating) """
        data = self.set.iloc[item]
        return self.all_users_index[data['userId']], self.all_items_index[data['movieId']], float(data['rating'])

    def __len__(self):
        return self.set.shape[0]

    @staticmethod
    def get_number_of_users():
        return len(MovieLensGNNDataset.all_users)

    @staticmethod
    def get_number_of_items():
        return len(MovieLensGNNDataset.all_items)
