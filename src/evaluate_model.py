import torch

from content_providers.dynamic_profiles_provider import DynamicProfilesProvider
from globals import test_set_file, val_batch_size
from neural_collaborative_filtering.datasets.dynamic_datasets import DynamicPointwiseDataset
from neural_collaborative_filtering.eval import eval_model
from neural_collaborative_filtering.models.attention_ncf import AttentionNCF
from neural_collaborative_filtering.util import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # parameters
    include_val_ratings_to_user_profiles = False
    model_file = '../models/runs/AttentionNCF_with_features_attNet128_3mil.pt'

    # load model (must be of the correct type)
    cp = DynamicProfilesProvider(include_val_ratings_to_user_profiles=include_val_ratings_to_user_profiles)
    test_dataset = DynamicPointwiseDataset(test_set_file, dynamic_provider=cp)
    model = load_model(model_file, AttentionNCF)
    print(model)

    # evaluate model on test set
    eval_model(model, test_dataset, val_batch_size, ranking=False, val_ratings_included=include_val_ratings_to_user_profiles)
