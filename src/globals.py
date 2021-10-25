imdb_path = '../data/imdb/'
movielens_path = '../data/movielens/'
rdf_path = '../data/rdf_data/'
audio_features_path = '../data/audio_features_movies/'

item_metadata_file = '../data/item_metadata'
user_ratings_file = '../data/user_ratings'
user_embeddings_file = '../data/user_embeddings'
train_set_file = '../data/train'
val_set_file = '../data/val'
test_set_file = '../data/test'

# random seed to use for data creation
seed = 102

""" NN hyperparameters """
checkpoint_model_path = '../models/checkpoint.pt'
final_model_path = '../models/final_model.pt'
max_epochs = 100
batch_size = 128
lr = 3e-4
weight_decay = 1e-5
dropout_rate = 0.5
early_stop = True
stop_with_train_loss_instead = False  # Note: useful if we are trying to overfit
patience = 3
