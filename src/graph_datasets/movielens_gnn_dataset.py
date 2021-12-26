import pandas as pd
import numpy as np
from torch_geometric.utils import to_networkx

from globals import user_ratings_file, train_set_file, val_set_file, test_set_file, \
    message_passing_vs_supervised_edges_ratio, item_metadata_file, movie_imdb_df_file, use_genre_nodes
from gnns.datasets.GNN_dataset import GNN_Dataset
from graph_datasets.graph_creation import create_onehot_graph, create_onehot_graph_from_utility_matrix
from plots import plot_user_item_graph
from recommendation.utility_matrix import UtilityMatrix


class MovieLensGNNDataset(GNN_Dataset):
    print('Initializing common dataset prerequisites ...')
    # user_ratings: pd.DataFrame = pd.read_hdf(user_ratings_file + '.h5')
    genres_df: pd.DataFrame = pd.read_csv(movie_imdb_df_file + '.csv', index_col='tconst')
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
            util_matrix = UtilityMatrix(train_set_file)
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
            util_matrix = UtilityMatrix(train_set_file)
        elif file == test_set_file:
            self.set = MovieLensGNNDataset.test_set
            self.graph_edges = MovieLensGNNDataset.train_set
            util_matrix = UtilityMatrix(train_set_file)
            # TODO: using val edges as well results in worse eval scores
            # self.graph_edges = MovieLensGNNDataset.train_set.append(MovieLensGNNDataset.val_set)  # use validation edges too
            # util_matrix = ???
        else:
            raise Exception('Invalid filepath for OneHot dataset')
        # create a corresponding graph depending on

        # self.known_graph, self.all_users_index, self.all_items_index = create_onehot_graph(MovieLensGNNDataset.all_users,
        #                                                                                    MovieLensGNNDataset.all_items,
        #                                                                                    self.graph_edges,
        #                                                                                    MovieLensGNNDataset.user_ratings)

        self.known_graph, self.all_users_index, self.all_items_index = create_onehot_graph_from_utility_matrix(
            util_matrix, genres=MovieLensGNNDataset.genres_df if use_genre_nodes else None
        )

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

    def get_class_counts(self):
        return self.set['rating'].value_counts()

    @staticmethod
    def get_number_of_users():
        return len(MovieLensGNNDataset.all_users)

    @staticmethod
    def get_number_of_items():
        return len(MovieLensGNNDataset.all_items)

    @staticmethod
    def get_initial_repr_dim():
        # For one-hot dataset:
        return MovieLensGNNDataset.get_number_of_items() + MovieLensGNNDataset.get_number_of_users()
