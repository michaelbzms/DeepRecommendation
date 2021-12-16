from torch.utils.data import DataLoader

from globals import train_set_file
from gnns.models.GCN import GCN_NCF
from graph_datasets.movielens_gnn_dataset import MovieLensGNNDataset


mld = MovieLensGNNDataset(train_set_file)

data_loader = DataLoader(mld, batch_size=16)

model = GCN_NCF()
model.forward(mld.known_graph, next(iter(data_loader)))

