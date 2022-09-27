from torch_geometric.data import Data


class ContentProvider:
    """ Extend this for specific features """
    def get_item_profile(self, itemID):
        raise NotImplementedError

    def get_user_profile(self, userID):
        raise NotImplementedError

    def get_num_items(self):
        raise NotImplementedError

    def get_num_users(self):
        raise NotImplementedError

    def get_item_feature_dim(self):
        raise NotImplementedError


class DynamicContentProvider:
    """ Extend this for specific features """
    def get_item_profile(self, itemID):
        raise NotImplementedError

    def get_num_items(self):
        raise NotImplementedError

    def get_num_users(self):
        raise NotImplementedError

    def get_item_feature_dim(self):
        raise NotImplementedError

    def collate_interacted_items(self, batch, for_ranking: bool):
        """
        Combine __getitem__() with this custom collate_fn to return for each batch in a data loader:
          > candidate_items_batch: (B, F)  B items with their features
          > rated_items_features: (I, F) I rated items with their features
          > user_matrix: (B, I) a subarray (not exactly) of the utility matrix with the (normalized) ratings of B users on I items.
        The order must match rated_items_feature's order on I axis.
        """
        raise NotImplementedError


class GraphContentProvider:
    """ Extend this for specific features and graphs """
    def get_num_items(self):
        raise NotImplementedError

    def get_num_users(self):
        raise NotImplementedError

    def get_user_nodeID(self, userID) -> int:
        raise NotImplementedError

    def get_item_nodeID(self, itemID) -> int:
        raise NotImplementedError

    def get_graph(self) -> Data:
        raise NotImplementedError     # returns graph (as in Data in PyG) reference
