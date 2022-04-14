import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from globals import full_matrix_file
from neural_collaborative_filtering.content_providers import GraphContentProvider


def create_graph(interactions: pd.DataFrame, node_feat, item_to_node_ID, user_to_node_ID):
    """ Assume node features where item nodes go first and then users (we might want to add more users) """
    # TODO: check these
    user_mean_ratings = interactions.groupby('userId')['rating'].mean()
    item_mean_ratings = interactions.groupby('movieId')['rating'].mean()
    # create edges and edge attributes
    edge_index = [[], []]
    edge_attr = []
    for _, (userId, itemId, rating) in tqdm(interactions.iterrows(),
                                            desc='Loading graph edges...',
                                            total=len(interactions)):
        # add edge user ----> item with weight: rating - avg_user_rating
        edge_index[0].append(user_to_node_ID[userId])
        edge_index[1].append(item_to_node_ID[itemId])
        edge_attr.append([rating - user_mean_ratings])
        # add edge item ----> user with weight: rating - avg_item_rating
        edge_index[0].append(item_to_node_ID[itemId])
        edge_index[1].append(user_to_node_ID[userId])
        edge_attr.append([rating - item_mean_ratings])
    # return Data object representing the graph
    return Data(
        x=node_feat,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )


class GraphProvider(GraphContentProvider):
    """ Base class for all graph providers where the type of initial node features can be changed via polymorphism."""
    def __init__(self, file):
        # load all known interactions (not just train/val/test)
        all_interactions = pd.read_csv(full_matrix_file + '.csv')
        # extract and sort all unique users and items that will become nodes
        self.all_items = sorted(all_interactions['movieId'].unique())
        self.all_users = sorted(all_interactions['userId'].unique())
        # assign each user and item to a node ID
        self.item_to_node_ID = {i: ind for ind, i in enumerate(self.all_items)}
        self.user_to_node_ID = {u: self.get_num_items() + ind for ind, u in enumerate(self.all_users)}

        # load interactions to use for the graph from the file
        self.interactions = pd.read_csv(file + '.csv')
        # one-hot vectors for node features
        node_features = self.get_node_features()
        # create graph
        self.graph = create_graph(self.interactions, node_features, self.item_to_node_ID, self.user_to_node_ID)

    def get_node_features(self):
        raise NotImplementedError

    def get_node_feature_dim(self):
        raise NotImplementedError

    def get_num_items(self):
        return len(self.all_items)

    def get_num_users(self):
        raise len(self.all_users)

    def get_user_nodeID(self, userID) -> int:
        return self.user_to_node_ID[userID]

    def get_item_nodeID(self, itemID) -> int:
        raise self.item_to_node_ID[itemID]

    def get_graph(self) -> Data:
        return self.graph


class OneHotGraphProvider(GraphProvider):
    """ A graph where the initial node embeddings are onehot vectors. """
    def get_node_features(self):
        return torch.eye(self.get_node_feature_dim())

    def get_node_feature_dim(self):
        return self.get_num_users() + self.get_num_items()


class ProfilesGraphProvider(GraphProvider):
    """ A graph where the initial node embeddings are content-based profiles """
    def __init__(self, file):
        super(ProfilesGraphProvider, self).__init__(file)

    def get_node_features(self):
        # TODO: load stuff here if not already, will be called in constructor, can add stuff to self here as well!
        return None

    def get_node_feature_dim(self):
        # TODO
        return None
