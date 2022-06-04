imdb_path = '../data/imdb/'
movielens_path = '../data/movielens/'
audio_features_path = '../data/audio_features_movies/'
plots_path = '../plots/'

data_folder = '../data'
item_metadata_file = '../data/item_metadata'
movie_imdb_df_file = '../data/movies_imdb_df'
movie_text_info_file = '../data/movie_info.csv'
user_ratings_file = '../data/user_ratings'
user_embeddings_file = '../data/user_embeddings'
audio_features_file = '../data/audio_features'
train_set_file = '../data/train'
val_set_file = '../data/val'
test_set_file = '../data/test'
ranking_train_set_file = '../data/ranking_train'
ranking_val_set_file = '../data/ranking_val'
ranking_test_set_file = '../data/ranking_test'
full_matrix_file = '../data/full_utility_matrix'

# random seed to use for data creation
seed = 102

# which features to use for items
""" 'metadata', 'audio' or 'all' """
features_to_use = 'metadata'

""" NN hyperparameters """
checkpoint_model_path = '../models/checkpoint.pt'
final_model_path = '../models/final_model.pt'
max_epochs = 40
batch_size = 256
val_batch_size = 512
lr = 3e-4  # 5e-4
weight_decay = 1e-5
dropout_rate = 0.2
early_stop = True
stop_with_train_loss_instead = False  # Note: useful if we are trying to overfit
patience = 3

# This doesn't help
mask_target_edges_when_training = False
message_passing_vs_supervised_edges_ratio = 0.7


USE_FEATURES = True
use_genre_nodes = False
