import torch

from neural_collaborative_filtering.content_providers import ContentProvider
from neural_collaborative_filtering.datasets.base import PointwiseDataset, RankingDataset
from neural_collaborative_filtering.models.base import NCF


class FixedPointwiseDataset(PointwiseDataset):
    def __init__(self, file: str, content_provider: ContentProvider):
        super().__init__(file)
        self.content_provider = content_provider

    def use_collate(self):          # much faster
        def custom_collate(batch, cp: ContentProvider):
            # turn per-row to per-column
            batch_data = list(zip(*batch))
            # get profiles in batches instead of one-by-one
            user_vecs = cp.get_user_profile(userID=batch_data[0])
            item_vecs = cp.get_item_profile(itemID=batch_data[1])
            return torch.FloatTensor(user_vecs), torch.FloatTensor(item_vecs), torch.FloatTensor(batch_data[2])

        return lambda batch: custom_collate(batch, self.content_provider)

    @staticmethod
    def do_forward(model: NCF, batch, device):
        # get the input matrices and the target
        user_vec, item_vec, y_batch = batch
        # forward model
        out = model(user_vec.float().to(device), item_vec.float().to(device))
        return out, y_batch


class FixedRankingDataset(RankingDataset):
    def __init__(self, ranking_file: str, content_provider: ContentProvider):
        super().__init__(ranking_file)
        self.content_provider = content_provider

    def use_collate(self):        # much faster this way
        def custom_collate(batch, cp: ContentProvider):
            # turn per-row to per-column
            batch_data = list(zip(*batch))
            # get profiles in batches instead of one-by-one
            user_vecs = cp.get_user_profile(userID=batch_data[0])
            item1_vecs = cp.get_item_profile(itemID=batch_data[1])
            item2_vecs = cp.get_item_profile(itemID=batch_data[2])
            return torch.FloatTensor(user_vecs), torch.FloatTensor(item1_vecs), torch.FloatTensor(item2_vecs)

        return lambda batch: custom_collate(batch, self.content_provider)

    @staticmethod
    def do_forward(model: NCF, batch, device):
        # get the input matrices and the target
        user_vec, item1_vec, item2_vec = batch
        # forward model
        out1 = model(user_vec.float().to(device), item1_vec.float().to(device))
        out2 = model(user_vec.float().to(device), item2_vec.float().to(device))
        return out1, out2
