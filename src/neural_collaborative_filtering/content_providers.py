

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
