import torch

from datasets.dynamic_movieLens_dataset import DynamicMovieLensDataset, MyCollator
from datasets.one_hot_dataset import OneHotMovieLensDataset
from globals import test_set_file, val_batch_size, USE_FEATURES
from neural_collaborative_filtering.evaluate import eval_model_with_visualization
from neural_collaborative_filtering.models.advanced_ncf import AttentionNCF
from neural_collaborative_filtering.models.basic_ncf import BasicMultimodalNCF
from neural_collaborative_filtering.models.basic_ncf import BasicNCF
from neural_collaborative_filtering.util import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False
keep_att_stats = False


if __name__ == '__main__':
    model_file = '../models/final_model.pt'

    if USE_FEATURES:
        dataset_class = DynamicMovieLensDataset

        # get metadata dim
        item_dim = dataset_class.get_item_feature_dim()

        # load model with correct layer sizes
        model = load_model(model_file, AttentionNCF)
    else:
        dataset_class = OneHotMovieLensDataset

        # model = load_model(model_file, BasicNCF)
        state, _ = torch.load(model_file)
        model = BasicMultimodalNCF(item_dim=dataset_class.get_number_of_items(),
                                   user_dim=dataset_class.get_number_of_users())
        model.load_state_dict(state)

        # make sure these are false
        visualize = False
        keep_att_stats = False

    print(model)

    # evaluate model on test set
    eval_model_with_visualization(model, dataset_class, test_set_file, val_batch_size)
