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
        raise NotImplementedError


class GraphContentProvider:
    """ Extend this for specific features and graphs """
    def get_num_items(self):
        raise NotImplementedError

    def get_num_users(self):
        raise NotImplementedError

    def get_node_feature_dim(self):
        raise NotImplementedError

    def get_user_nodeID(self, userID) -> int:
        raise NotImplementedError

    def get_item_nodeID(self, itemID) -> int:
        raise NotImplementedError

    def get_graph(self) -> Data:
        raise NotImplementedError     # returns graph (as in Data in PyG) reference

    # TODO: methods to add more nodes (e.g. user nodes for inference on new users)?
