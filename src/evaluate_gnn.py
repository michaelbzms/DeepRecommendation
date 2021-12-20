import torch

from globals import test_set_file, val_batch_size
from gnns.evaluate_gnn_ncf import eval_model
from gnns.models.GAT import GAT_NCF
from gnns.models.GNN import load_gnn_model
from graph_datasets.movielens_gnn_dataset import MovieLensGNNDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# perform attention visualization on top of evaluation
visualize = False
keep_att_stats = False


if __name__ == '__main__':
    model_file = '../models/GAT_0491val.pt'

    dataset_class = MovieLensGNNDataset
    initial_repr_dim = dataset_class.get_initial_repr_dim()

    model = load_gnn_model(model_file, GAT_NCF, initial_repr_dim)
    print(model)

    # evaluate model on test set
    eval_model(model, dataset_class, test_set_file, val_batch_size)
