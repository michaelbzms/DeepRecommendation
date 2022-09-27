# file paths
plots_path = '../plots/'

imdb_path = '../data/imdb/'
movielens_path = '../data/movielens/'
movie_text_info_file = '../data/movie_info.csv'
movie_imdb_df_file = '../data/movies_imdb_df'

item_metadata_file = '../data/item_metadata'
audio_features_path = '../data/audio_features_movies/'
audio_features_file = '../data/audio_features'

user_ratings_file = '../data/user_ratings'
user_embeddings_file = '../data/user_embeddings'
user_ratings_with_val_file = '../data/user_ratings_with_val'
user_embeddings_with_val_file = '../data/user_embeddings_with_val'

train_set_file = '../data/train'
train_and_val_set_file = '../data/train_and_val'
ranking_train_set_file = '../data/ranking_train'
val_set_file = '../data/val'
test_set_file = '../data/test'
full_matrix_file = '../data/full_utility_matrix'

# random seed to use for reproducible data creation
seed = 42

# parallelism on dataloaders on training
num_workers = 0    # Note: Currently can't do more workers than 0 because collate functions cannot be pickled

# NN hyperparameters (some are global some are set locally)
checkpoint_model_path = '../models/checkpoint.pt'
final_model_path = '../models/final_model.pt'
max_epochs = 32
patience = 3
val_batch_size = 512
