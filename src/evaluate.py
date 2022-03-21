import torch

from content_providers.fixed_profiles_provider import FixedProfilesProvider
from content_providers.one_hot_provider import OneHotProvider
from globals import test_set_file, val_batch_size, USE_FEATURES
from neural_collaborative_filtering.datasets.fixed_datasets import PointwiseDataset
from neural_collaborative_filtering.eval import eval_model
from neural_collaborative_filtering.models.advanced_ncf import AttentionNCF
from neural_collaborative_filtering.models.basic_ncf import BasicMultimodalNCF
from neural_collaborative_filtering.models.basic_ncf import BasicNCF
from neural_collaborative_filtering.util import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False
keep_att_stats = False


if __name__ == '__main__':
    model_file = '../models/checkpoint.pt'

    if USE_FEATURES:
        # dataset_class = DynamicMovieLensDataset
        #
        # # get metadata dim
        # item_dim = dataset_class.get_item_feature_dim()
        #
        # # load model with correct layer sizes
        # model = load_model(model_file, AttentionNCF)

        fixed_provider = FixedProfilesProvider()
        test_dataset = PointwiseDataset(test_set_file, content_provider=fixed_provider)

        # random model
        # model = BasicNCF(item_dim=fixed_provider.get_item_feature_dim(),
        #                  user_dim=fixed_provider.get_item_feature_dim())

        model = load_model(model_file, BasicNCF)
    else:
        onehot_provider = OneHotProvider()
        test_dataset = PointwiseDataset(test_set_file, content_provider=onehot_provider)

        model = load_model(model_file, BasicNCF)
        # state, _ = torch.load(model_file)
        # model = BasicNCF(item_dim=onehot_provider.get_num_items(),
        #                  user_dim=onehot_provider.get_num_users())
        # model.load_state_dict(state)

        # make sure these are false
        visualize = False
        keep_att_stats = False

    print(model)

    # evaluate model on test set
    eval_model(model, test_dataset, val_batch_size)
