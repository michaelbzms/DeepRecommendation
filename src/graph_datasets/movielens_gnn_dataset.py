import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_networkx
from tqdm import tqdm

from globals import user_ratings_file, train_set_file, val_set_file, test_set_file, \
    message_passing_vs_supervised_edges_ratio
from gnns.datasets.GNN_dataset import GNN_Dataset
from plots import plot_user_item_graph


def create_onehot_graph(all_users: np.array, all_items: np.array, graph_edges, user_ratings):
    """ Create the graph from user and item nodes and use given edges with weights between them """
    print('Creating graph...')
    # First sorted items then sorted users as nodes with combined one-hot vector representations
    x = torch.eye(len(all_items) + len(all_users))

    all_items_index = {i: ind for ind, i in enumerate(all_items)}
    all_users_index = {u: ind + len(all_items) for ind, u in enumerate(all_users)}       # IMPORTANT: add num items to user index!!!

    # find edges
    edge_index = [[all_users_index[u] for u in graph_edges['userId']],
                  [all_items_index[i] for i in graph_edges['movieId']]]
    # append backward edges too
    edge_index[0] += edge_index[1]
    edge_index[1] += edge_index[0][:graph_edges.shape[0]]

    # TODO: This is dumb
    # edge_attr = [[rating] for rating in graph_edges['rating']] * 2

    # Note: use rating - avg user rating instead of just the rating. Sign is meaningful this way
    # Note: Negative weights give nan values. Why??? -> because of sqrt(node_degree). Setting normalize=False fixes it
    edge_attr = [[(edge['rating'] - user_ratings.loc[int(edge['userId'])]['meanRating'])]
                 for _, edge in tqdm(graph_edges.iterrows(), desc='Loading graph edges...', total=len(graph_edges))] * 2

    # edge_index = [[all_users_index[edge['userId']] for _, edge in graph_edges.iterrows()
    #                if edge['rating'] > user_ratings.loc[int(edge['userId'])]['meanRating']],
    #               [all_items_index[edge['movieId']] for _, edge in graph_edges.iterrows()
    #                if edge['rating'] > user_ratings.loc[int(edge['userId'])]['meanRating']]]
    # # append backward edges
    # start_len = len(edge_index[0])
    # edge_index[0] += edge_index[1]
    # edge_index[1] += edge_index[0][:start_len]
    #
    # # Add only edges that are higher that the user's average ratings
    # edge_attr = [[-edge['rating']]
    #              for _, edge in tqdm(graph_edges.iterrows(), desc='Loading graph edges...', total=len(graph_edges))
    #              if edge['rating'] > user_ratings.loc[int(edge['userId'])]['meanRating']] * 2

    known_graph = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        num_items=len(all_items)
    )
    print(known_graph)

    # remove duplicates
    print('Removing duplicate edges...')
    known_graph.edge_index, known_graph.edge_attr = coalesce(known_graph.edge_index, known_graph.edge_attr, reduce='mean')
    print(known_graph)
    print('done.')

    return known_graph, all_users_index, all_items_index


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

    def __init__(self, file: str, mask_target_edges_when_training=False):
        self.mask_target_edges_when_training = mask_target_edges_when_training
        if file == train_set_file:
            if not mask_target_edges_when_training:
                self.set = MovieLensGNNDataset.train_set
                self.graph_edges = MovieLensGNNDataset.train_set
            else:
                # graph egdes are disjoint from train set -> split training edges to message passing ones and supervision edges
                r = message_passing_vs_supervised_edges_ratio
                # randomly select r % of each user's ratings to be used for graph edges
                self.graph_edges = MovieLensGNNDataset.train_set.groupby('userId', as_index=False) \
                    .apply(lambda obj: obj.loc[np.random.choice(obj.index, int(r * obj.shape[0]), replace=False), :]).reset_index(drop=True)
                # and use the rest as supervised edges by removing the graph edges from the train set
                self.set = pd.merge(MovieLensGNNDataset.train_set, self.graph_edges, on=['userId', 'movieId', 'rating'], how='outer', indicator=True) \
                    .query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)

        elif file == val_set_file:
            self.set = MovieLensGNNDataset.val_set
            self.graph_edges = MovieLensGNNDataset.train_set
        elif file == test_set_file:
            self.set = MovieLensGNNDataset.test_set
            self.graph_edges = MovieLensGNNDataset.train_set.append(MovieLensGNNDataset.val_set)  # use validation edges too
        else:
            raise Exception('Invalid filepath for OneHot dataset')
        # create a corresponding graph depending on
        self.known_graph, self.all_users_index, self.all_items_index = create_onehot_graph(MovieLensGNNDataset.all_users,
                                                                                           MovieLensGNNDataset.all_items,
                                                                                           self.graph_edges,
                                                                                           MovieLensGNNDataset.user_ratings)

    def __getitem__(self, item):
        """ returns (user_index, item_index, target rating) """
        data = self.set.iloc[item]
        return int(self.all_users_index[data['userId']]), int(self.all_items_index[data['movieId']]), float(data['rating'])

    def get_graph(self):
        return self.known_graph

    def draw_graph(self):
        g = to_networkx(self.known_graph, to_undirected=True)
                        # node_attrs=list(self.all_items) + list(self.all_users),
                        # edge_attrs=[str(w) for w in self.known_graph.edge_attr.numpy()]
        plot_user_item_graph(g)

    def __len__(self):
        return self.set.shape[0]

    @staticmethod
    def get_number_of_users():
        return len(MovieLensGNNDataset.all_users)

    @staticmethod
    def get_number_of_items():
        return len(MovieLensGNNDataset.all_items)
