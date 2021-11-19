imdb_path = '../data/imdb/'
movielens_path = '../data/movielens/'
rdf_path = '../data/rdf_data/'
audio_features_path = '../data/audio_features_movies/'
plots_path = '../plots/'

item_metadata_file = '../data/item_metadata'
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
max_epochs = 100
batch_size = 128
val_batch_size = 128
embeddings_lr = 3e-4
lr = 3e-4   # or 1e-3 if taking too long
weight_decay = 1e-5
dropout_rate = 0.2
early_stop = True
stop_with_train_loss_instead = False  # Note: useful if we are trying to overfit
patience = 5
