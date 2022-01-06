imdb_path = '../data/imdb/'
movielens_path = '../data/movielens/'
audio_features_path = '../data/audio_features_movies/'
plots_path = '../plots/'

data_folder = '../data'
item_metadata_file = '../data/item_metadata'
movie_imdb_df_file = '../data/movies_imdb_df'
user_ratings_file = '../data/user_ratings'
user_embeddings_file = '../data/user_embeddings'
audio_features_file = '../data/audio_features'
train_set_file = '../data/train'
val_set_file = '../data/val'
test_set_file = '../data/test'
full_matrix_file = '../data/full_utility_matrix'

# random seed to use for data creation
seed = 102

# which features to use for items
""" 'metadata', 'audio' or 'all' """
features_to_use = 'all'

""" NN hyperparameters """
checkpoint_model_path = '../models/checkpoint.pt'
final_model_path = '../models/final_model.pt'
max_epochs = 256
batch_size = 256  # 16384
val_batch_size = 256
embeddings_lr = 0.001   # not currently used
lr = 1e-3  # 5e-4
weight_decay = 1e-5
dropout_rate = 0.2
early_stop = True
stop_with_train_loss_instead = False  # Note: useful if we are trying to overfit
patience = 5

# This doesn't help
mask_target_edges_when_training = False
message_passing_vs_supervised_edges_ratio = 0.7

# Try weighted MSE loss
use_weighted_mse_for_training = False

USE_FEATURES = False
use_genre_nodes = True
