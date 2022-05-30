import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from globals import full_matrix_file, item_metadata_file, user_embeddings_file
from neural_collaborative_filtering.content_providers import GraphContentProvider


def create_graph(interactions: pd.DataFrame, item_features, user_features, item_to_node_ID, user_to_node_ID, binary=False):   # TODO: must be false if we are masking later
    """ Assume node features where item nodes go first and then users (we might want to add more users) """
    # calculate mean ratings per user and per item as an edge attribute
    user_mean_ratings = interactions.groupby('userId')['rating'].mean()
    item_mean_ratings = interactions.groupby('movieId')['rating'].mean()
    # create edges and edge attributes
    edge_index = [[], []]
    edge_attr = []
    pos = []
    i = 0
    for _, (userId, itemId, rating) in tqdm(interactions.iterrows(),
                                            desc='Loading graph edges...',
                                            total=len(interactions)):
        # convert to Node IDs
        userNodeId = user_to_node_ID[userId]
        itemNodeId = item_to_node_ID[itemId]

        # TODO: which should we use?
        # TODO: message dropout does not account for the avg used in the edge_attr so we are kind of cheating there...
        # add edge user ----> item with weight: rating - avg_user_rating
        user_avg = (user_mean_ratings.loc[userId] + 2.5) / 2
        if not binary or rating >= user_avg:
            edge_index[0].append(userNodeId)
            edge_index[1].append(itemNodeId)
            if not binary:
                edge_attr.append(rating - user_avg)
            pos.append([userNodeId, itemNodeId, i])
            i += 1

        # add edge item ----> user with weight: rating - avg_item_rating
        item_avg = (item_mean_ratings.loc[itemId] + 2.5) / 2
        if not binary or rating >= item_avg:
            edge_index[0].append(itemNodeId)
            edge_index[1].append(userNodeId)
            if not binary:
                edge_attr.append(rating - item_avg)
            pos.append([itemNodeId, userNodeId, i])
            i += 1
    print(f'Used {len(edge_index[0])} (directed) edges ({len(edge_index[0]) / (2 * len(interactions)) * 100.0:.2f}% of all possible) for graph.')
    # create mult index df with the positions of each edge
    pos_df = pd.DataFrame(pos, columns=['Id1', 'Id2', 'pos']).set_index(['Id1', 'Id2'], inplace=False)
    print('Created pos multi-index df.')
    # return Data object representing the graph
    return Data(
        item_features=item_features,
        user_features=user_features,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float) if not binary else None,
        pos_df=pos_df
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
        # node features
        item_features = self.get_item_features()
        user_features = self.get_user_features()
        # create graph
        self.graph = create_graph(self.interactions, item_features, user_features, self.item_to_node_ID, self.user_to_node_ID)

    def get_item_features(self):
        raise NotImplementedError

    def get_user_features(self):
        raise NotImplementedError

    def get_node_features(self):
        # Note: not used currently in favor of always having separate item and user features
        raise NotImplementedError

    def get_num_items(self):
        return len(self.all_items)

    def get_num_users(self):
        return len(self.all_users)

    def get_item_dim(self):
        raise NotImplementedError

    def get_user_dim(self):
        raise NotImplementedError

    def get_user_nodeID(self, userID) -> int:
        return self.user_to_node_ID[userID]

    def get_item_nodeID(self, itemID) -> int:
        return self.item_to_node_ID[itemID]

    def get_graph(self) -> Data:
        return self.graph


class OneHotGraphProvider(GraphProvider):
    """ A graph where the initial node embeddings are onehot vectors. """
    def get_item_features(self):
        return torch.eye(self.get_num_items())

    def get_user_features(self):
        return torch.eye(self.get_num_users())

    def get_item_dim(self):
        return self.get_num_items()

    def get_user_dim(self):
        return self.get_num_users()

    def get_node_features(self):   # not used
        return torch.eye(self.get_num_items() + self.get_num_items())


class ProfilesGraphProvider(GraphProvider):
    """ A graph where the initial node embeddings are content-based profiles """
    def __init__(self, file):
        self.metadata: pd.DataFrame = pd.read_hdf(item_metadata_file + '.h5')
        self.user_embeddings: pd.DataFrame = pd.read_hdf(user_embeddings_file + '.h5')
        super(ProfilesGraphProvider, self).__init__(file)

    def get_item_features(self):
        item_profiles = self.metadata.loc[self.all_items, :].values
        return torch.FloatTensor(item_profiles)

    def get_user_features(self):
        user_profiles = self.user_embeddings.loc[self.all_users, :].values
        return torch.FloatTensor(user_profiles)

    def get_item_dim(self):
        return self.metadata.shape[1]

    def get_user_dim(self):
        return self.user_embeddings.shape[1]

    def get_node_features(self):     # not used
        # get item and user profiles
        item_profiles = self.get_item_features()
        user_profiles = self.get_user_features()
        # concat them into node features
        return torch.vstack([item_profiles, user_profiles])
