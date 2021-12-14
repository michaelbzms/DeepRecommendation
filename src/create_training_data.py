import pandas as pd
import numpy as np
from tqdm import tqdm

from globals import movielens_path, item_metadata_file, train_set_file, val_set_file, test_set_file, seed, \
    user_ratings_file, user_embeddings_file, full_matrix_file, imdb_path
from util import multi_hot_encode


def load_user_ratings(movielens_data_folder, LIMIT_USERS=None):
    # load movielens user reviews data
    user_ratings = pd.read_csv(movielens_data_folder + 'ratings.csv',
                               index_col='userId',
                               usecols=['userId', 'movieId', 'rating', 'timestamp'],
                               dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
    if LIMIT_USERS is not None:
        print('Limiting number of users to', LIMIT_USERS)
        # user_ratings = user_ratings.loc[1: LIMIT_USERS]
        user_ratings = user_ratings.loc[user_ratings.index.max() - LIMIT_USERS: user_ratings.index.max()]

    # load genome tags
    genometags = pd.read_csv(movielens_data_folder + 'genome-scores.csv',
                             index_col='movieId',
                             usecols=['movieId', 'tagId', 'relevance'],
                             dtype={'movieId': np.int32, 'tagId': np.int32, 'relevance': np.float64})
    genometags = genometags.pivot_table('relevance', index='movieId', columns='tagId')

    # change movieId to IMDb ID, link movieIds with imdbIds
    links = pd.read_csv(movielens_data_folder + 'links.csv',
                        index_col='movieId',
                        usecols=['movieId', 'imdbId'],
                        dtype={'movieId': np.int32, 'imdbId': 'string'})
    user_ratings['movieId'] = 'tt' + user_ratings['movieId'].map(links['imdbId'])
    genometags.index = 'tt' + genometags.index.map(links['imdbId'])
    genometags.rename({c: f'genome_{c}' for c in genometags.columns.tolist()}, inplace=True, axis=1)   # needed for concat to work later

    return user_ratings, genometags


def load_imdb_dfs(unique_movies: pd.Series):
    # load each file with tconst (title id) as index
    print('Loading IMDB data...')
    tconst_files = [
        ('title.basics.tsv', None),
        ('title.ratings.tsv', None)
    ]
    all_dfs = []
    for file, usecols in tconst_files:
        df = pd.read_csv(imdb_path + file, index_col='tconst',  usecols=usecols,
                         sep='\t', encoding='utf-8',
                         keep_default_na=False, na_values=['\\N'])
        all_dfs.append(df)

    # combine all into one big fat DataFrame
    print('concatenating...')
    movies_df = pd.concat(all_dfs, axis=1)
    print('Reducing size to movies given...')
    movies_df = movies_df[(movies_df['titleType'].isin(['tvMovie', 'movie'])) &
                          (movies_df.index.isin(unique_movies.index))]
    print('done')

    # fix NA and types afterwards as it is not supported for read_csv
    movies_df['numVotes'] = movies_df['numVotes'].fillna(0).astype(np.uint16)
    movies_df['isAdult'] = movies_df['isAdult'].astype(bool)
    movies_df['startYear'] = movies_df['startYear'].fillna(0).astype(np.uint16)
    movies_df['endYear'] = movies_df['endYear'].fillna(0).astype(np.uint16)
    movies_df['genres'] = movies_df['genres'].fillna('').astype(str)

    # filtering
    # movies_df = movies_df[(movies_df['numVotes'] >= MIN_VOTES) &
    #                       (~(movies_df['genres'].str.contains('Short', regex=False, na=False))) &
    #                       (movies_df['genres'].str != '')]

    print('Loading edges')
    principals_df = pd.read_csv(imdb_path + 'title.principals.tsv',
                                sep='\t',
                                encoding='utf-8',
                                keep_default_na=False,
                                na_values=['\\N'],
                                index_col='tconst',
                                usecols=['tconst', 'nconst', 'category'])
    principals_df = principals_df[principals_df.index.isin(movies_df.index)]
    principals_df = principals_df[principals_df['category'].isin(['actor', 'actress', 'writer', 'director', 'composer'])]

    print(movies_df)
    print(movies_df.shape)
    print(principals_df)
    print(principals_df.shape)

    return movies_df, principals_df


def load_imdb_metadata_features(unique_movies: pd.Series, MIN_APPEARANCES=1):
    movies_df, principals_df = load_imdb_dfs(unique_movies)

    all_genres = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',  'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',  'Thriller', 'War', 'Western', 'Biography', 'Music'
        # Rare (for now) categories: 'History', 'Family', 'Sport'
    ]

    # Don't actually need to do this separately:
    # actors_mask = (principals_df['category'] == 'actor') | (principals_df['category'] == 'actress')
    # all_actors = sorted(list(principals_df['nconst'][actors_mask].unique()))
    #
    # directors_mask = principals_df['category'] == 'director'
    # all_directors = sorted(list(principals_df['nconst'][directors_mask].unique()))
    #
    # composer_mask = principals_df['category'] == 'composer'
    # all_composers = sorted(list(principals_df['nconst'][composer_mask].unique()))

    print('Number of personel:', len(principals_df), f'. Removing those with less than {MIN_APPEARANCES} appearances...')
    reduced_principals_df = principals_df.groupby('nconst').filter(lambda x: len(x) >= MIN_APPEARANCES)
    print(f'Left with {len(reduced_principals_df)}')

    all_personnel = sorted(list(reduced_principals_df['nconst'].unique()))

    F = len(all_genres) + len(all_personnel)
    features = np.zeros((len(unique_movies), F))
    for i, movieId in tqdm(enumerate(unique_movies.index), total=len(unique_movies)):
        # multi-hot encode genres
        genres = movies_df.loc[movieId]['genres'].split(',')
        genres = set([g.replace(' ', '') for g in genres])
        genres_feat = multi_hot_encode([genres], all_genres)
        # multi-hot encode personel
        personnel = set(principals_df.loc[movieId]['nconst'])
        personnel_feat = multi_hot_encode([personnel], all_personnel)
        # put together for features
        features[i, :len(all_genres)] = genres_feat
        features[i, len(all_genres):] = personnel_feat
    return pd.DataFrame(index=unique_movies.index, data=features)


def save_set(matrix: pd.DataFrame, name: str):
    matrix.to_csv(name + '.csv', columns=['movieId', 'rating'], mode='w')


if __name__ == '__main__':
    recalculate_metadata = True
    save_user_ratings = True
    use_genome_tags = True
    random_vs_temporal_splitting = True
    create_user_embeddings_too = True
    split_embeddings_from_train = False  # don't do this
    use_audio = True
    LIMIT_USERS = None
    MIN_VOTES = 100  # 70

    # load user ratings (sparse representation of a utility matrix)
    print('Loading movieLens data...')
    utility_matrix, genome_metadata = load_user_ratings(movielens_path, LIMIT_USERS=LIMIT_USERS)
    print(utility_matrix.shape)

    # load audio features
    if use_audio:
        audio_features = pd.read_csv('../data/audio_features.csv', index_col='movieId', sep=';')
        utility_matrix = utility_matrix[utility_matrix['movieId'].isin(audio_features.index)]
        # utility_matrix = audio_features.join(utility_matrix, on='movieId', how='inner')
        print(utility_matrix)
        print(utility_matrix.shape)
        # filter utility matrix as per users:
        user_votes = utility_matrix.groupby('userId')['rating'].count()
        print('Original users:', len(user_votes))
        user_votes = user_votes[user_votes >= MIN_VOTES]  # at least these many votes on movies
        print('Keeping this many users based on number of votes:', len(user_votes))
        utility_matrix = utility_matrix[utility_matrix.index.isin(user_votes.index)]
        print('Utility matrix:', utility_matrix.shape)
        # utility_matrix['rating'].hist()

    # load movie features from only for movies in movieLens (for which we have ratings)
    if recalculate_metadata:
        print('Loading IMDb data...')
        unique_movies = pd.Series(index=utility_matrix['movieId'].unique().copy())
        imdb_metadata = load_imdb_metadata_features(unique_movies)
        if use_genome_tags:
            genome_metadata = genome_metadata[genome_metadata.index.isin(unique_movies.index)]
            metadata = imdb_metadata.join(genome_metadata)
            metadata = metadata.fillna(0.0)   # shouldn't be any but just in case
        else:
            metadata = imdb_metadata
        print(f'Found {metadata.shape[0]} movies.\nSaving metadata...')
        print(metadata)
        print(metadata.columns.tolist())
        metadata.to_hdf(item_metadata_file + '.h5', key='metadata', mode='w')
        print('OK!')
    else:
        metadata = pd.read_hdf(item_metadata_file + '.h5', key='metadata')
    # Note to check statistics: metadata.sum(axis=0)
    print(metadata.shape)
    print('Statistics:')
    print(metadata.sum(axis=0))

    # Note: there can still be movies in ratings for which we have no features
    # so remove them like this:
    print('Removing movies for which we have no features...')
    utility_matrix = utility_matrix[utility_matrix['movieId'].isin(metadata.index)]
    print('Final # of movies:', len(utility_matrix['movieId'].unique()))

    # train-val-test split (global temporal splitting)
    print('Calculating train-val-test split...')
    if random_vs_temporal_splitting:
        size: int = len(utility_matrix)
        val_split = int(np.floor(0.15 * size))
        test_split = val_split + int(np.floor(0.15 * size))
        indices = list(range(size))
        np.random.seed(seed)
        np.random.shuffle(indices)
        val = utility_matrix.iloc[indices[:val_split]]
        test = utility_matrix.iloc[indices[val_split: test_split]]
        if split_embeddings_from_train:  # bad idea, isn't helping
            embedding_split = test_split + int(np.floor(0.35 * size))  # (1 - (0.15 + 0.15)) / 2 = 0.7 / 2 = 0.4
            embeddings = utility_matrix.iloc[indices[test_split: embedding_split]]
            train = utility_matrix.iloc[indices[embedding_split:]]
        else:
            train = utility_matrix.iloc[indices[test_split:]]
            embeddings = None
    else:
        val_size = 0.15
        test_size = 0.15
        # calculate proper timestamp splits for each user based on his ratings
        val_splits = utility_matrix['timestamp'].groupby('userId').quantile(1.0 - val_size - test_size).astype(int)
        test_splits = utility_matrix['timestamp'].groupby('userId').quantile(1.0 - test_size).astype(int)
        # broadcast the result properly in the utility matrix
        utility_matrix['val_split'] = val_splits
        utility_matrix['test_split'] = test_splits
        # do train-val-test split according to the new broadcasted columns
        train = utility_matrix[utility_matrix['timestamp'] < utility_matrix['val_split']]
        val = utility_matrix[(utility_matrix['timestamp'] >= utility_matrix['val_split']) & (
                    utility_matrix['timestamp'] < utility_matrix['test_split'])]
        test = utility_matrix[utility_matrix['timestamp'] >= utility_matrix['test_split']]
        embeddings = None

    print(f'Training shape: {train.shape}, Validation shape: {val.shape}, Test shape: {test.shape}' + (
        f', Embedding shape: {embeddings.shape}' if split_embeddings_from_train else ''))

    if save_user_ratings:
        # user_ratings: pd.DataFrame = utility_matrix.drop('timestamp', axis=1).groupby('userId').apply(list)
        print('Saving user ratings from train set only...')
        if split_embeddings_from_train:
            ratings_to_use = embeddings
        else:
            ratings_to_use = train
        # IMPORTANT to sort by movieId
        user_ratings: pd.DataFrame = ratings_to_use.drop('timestamp', axis=1).sort_values(by='movieId').groupby(
            'userId').agg({'rating': list, 'movieId': list})
        user_ratings['rating'] = user_ratings['rating'].apply(lambda x: np.array(x))
        user_ratings['movieId'] = user_ratings['movieId'].apply(lambda x: np.array(x))
        user_ratings['meanRating'] = user_ratings['rating'].apply(lambda x: np.mean(x))
        user_ratings['numRatings'] = user_ratings['rating'].apply(lambda x: len(x))
        print('User ratings:\n', user_ratings)
        print(user_ratings.shape)
        user_ratings.to_hdf(user_ratings_file + '.h5', key='user_ratings', mode='w')
        print('OK!')
        print(f'Average number of ratings per user (in train set): {user_ratings["rating"].apply(lambda x: len(x)).mean()}')

        if create_user_embeddings_too:
            # create user_embeddings from user ratings once beforehand
            # Note: This takes a very long time
            print('Creating user embeddings...')


            def create_user_embedding(user_ratings: pd.DataFrame, metadata: pd.DataFrame):
                avg_rating = user_ratings['rating'].mean()
                return ((user_ratings['rating'] - avg_rating).reshape(-1, 1) * metadata.loc[user_ratings['movieId']].values).mean()  # TODO: sum or mean?


            user_embeddings = pd.DataFrame(index=user_ratings.index.unique().copy(), data={'embedding': object})
            for userId, user_ratings in tqdm(user_ratings.groupby('userId'), desc='Creating user embeddings...'):
                # Note: iloc[0] is needed because of some weird encapsulation idk
                user_embeddings.at[userId, 'embedding'] = create_user_embedding(user_ratings.iloc[0], metadata)
            print('Saving...')
            user_embeddings.to_hdf(user_embeddings_file + '.h5', key='user_embeddings', mode='w')
            print('Done')

    print('Saving sets...')
    save_set(train, train_set_file)
    save_set(val, val_set_file)
    save_set(test, test_set_file)
    save_set(utility_matrix, full_matrix_file)
    print('OK!')

    train.hist(column='timestamp')
    val.hist(column='timestamp')
    test.hist(column='timestamp')
