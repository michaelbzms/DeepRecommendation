import torch

from globals import test_set_file, val_batch_size
from neural_collaborative_filtering.evaluate import eval_model
from neural_collaborative_filtering.models.gnn_ncf import GAT_NCF
from neural_collaborative_filtering.util import load_model
from graph_datasets.movielens_gnn_dataset import MovieLensGNNDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False
keep_att_stats = False


if __name__ == '__main__':
    model_file = '../models/final_model.pt'

    dataset_class = MovieLensGNNDataset
    initial_repr_dim = dataset_class.get_initial_repr_dim()
    edge_dim = 2        # TODO

    model = load_model(model_file, GAT_NCF, initial_repr_dim=initial_repr_dim, edge_dim=edge_dim)
    print(model)

    # evaluate model on test set
    eval_model(model, dataset_class(test_set_file), val_batch_size)
