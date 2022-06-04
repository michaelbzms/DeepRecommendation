import torch

from content_providers.dynamic_profiles_provider import DynamicProfilesProvider
from content_providers.fixed_profiles_provider import FixedProfilesProvider
from content_providers.one_hot_provider import OneHotProvider
from globals import test_set_file, val_batch_size, USE_FEATURES
from neural_collaborative_filtering.datasets.dynamic_datasets import DynamicPointwiseDataset
from neural_collaborative_filtering.datasets.fixed_datasets import FixedPointwiseDataset
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
    # TODO: this setup is horrible

    model_file = '../models/runs/AttentionNCF_with_features_attNet128.pt'

    if USE_FEATURES:
        # dataset_class = DynamicMovieLensDataset
        #
        # # get metadata dim
        # item_dim = dataset_class.get_item_feature_dim()
        #
        # # load model with correct layer sizes
        # model = load_model(model_file, AttentionNCF)

        cp = DynamicProfilesProvider()
        test_dataset = DynamicPointwiseDataset(test_set_file, dynamic_provider=cp)

        model = load_model(model_file, AttentionNCF)
    else:
        onehot_provider = OneHotProvider()
        test_dataset = FixedPointwiseDataset(test_set_file, content_provider=onehot_provider)

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
    eval_model(model, test_dataset, val_batch_size, ranking=False)
