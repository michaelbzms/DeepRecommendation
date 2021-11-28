import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from rdflib import Graph, Namespace
from tqdm import tqdm
import warnings

from globals import movielens_path, rdf_path, item_metadata_file, train_set_file, val_set_file, test_set_file, seed, \
    user_ratings_file, user_embeddings_file, full_matrix_file


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

    return user_ratings, genometags


def extract_binary_features(actual_values: set, ordered_possible_values: list) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform([actual_values])
    return binary_format


def load_imdb_metadata_features(unique_movies: pd.Series, use_extended=False):
    # NAMESPACES
    ns_movies = Namespace('https://www.imdb.com/title/')
    ns_genres = Namespace('https://www.imdb.com/search/title/?genres=')
    ns_principals = Namespace('https://www.imdb.com/name/')
    ns_predicates = Namespace('http://example.org/props/')

    # our_movies = '("' + '"), ("'.join(unique_movies.index) + '")'
    # our_movies = '<' + '> <'.join(unique_movies.index) + '>'

    print('Loading rdf...')
    rdf = Graph().parse(rdf_path + ('movies_extended.nt' if use_extended else 'movies_pruned_actors.nt'), format='nt')
    print('done')

    # get all possible categorical features
    print('Looking up genres...')
    all_genres = rdf.query(
        """ SELECT DISTINCT ?genre
            WHERE {
                ?movie pred:hasGenre ?genre . 
            }""", initNs={'pred': ns_predicates})
    all_genres = sorted([str(g['genre']) for g in all_genres if len(str(g['genre']).split('=')[-1]) > 0])
    print('Found', len(all_genres), 'genres.')

    print('Looking up actors...')
    LEAST_MOVIES = 5  # Ignore insignificant actors
    all_actors = rdf.query(
        """ SELECT DISTINCT ?actor
            WHERE {
                ?movie pred:hasActor ?actor . 
            } 
            GROUP BY ?actor 
            HAVING (COUNT(?movie) >= """ + str(LEAST_MOVIES) + ')',
        initNs={'pred': ns_predicates})
    all_actors = sorted([str(a['actor']) for a in all_actors])
    # Note: keep just the id with: actors = sorted([str(a['actor']).split('/')[-1] for a in actors])
    print('Found', len(all_actors), 'actors with at least', LEAST_MOVIES, 'movies made.')

    print('Looking up directors...')
    LEAST_MOVIES2 = 7
    all_directors = rdf.query(
        """ SELECT DISTINCT ?director
            WHERE {
                ?movie pred:hasDirector ?director . 
            }
            GROUP BY ?director
            HAVING (COUNT(?movie) >= """ + str(LEAST_MOVIES2) + ')',
        initNs={'pred': ns_predicates})
    all_directors = sorted([str(d['director']) for d in all_directors])
    print('Found', len(all_directors), 'directors with at least', LEAST_MOVIES2, 'movies directed.')

    # Query all movies on rdf and their associated features
    print('Querying movie features...')
    movies = rdf.query(
        """SELECT DISTINCT ?movie ?year ?rating
              (group_concat(distinct ?genre; separator=",") as ?genres)
              (group_concat(distinct ?actor; separator=",") as ?actors)
              (group_concat(distinct ?director; separator=",") as ?directors)
           WHERE { 
              ?movie pred:hasYear ?year .
              ?movie pred:hasRating ?rating .
              ?movie pred:hasGenre ?genre .
              ?movie pred:hasDirector ?director .
              ?movie pred:hasActor ?actor .
           } 
           GROUP BY ?movie ?year ?rating""",
        initNs={'movies': ns_movies,
                'genres': ns_genres,
                'pred': ns_predicates,
                'principals': ns_principals})
    print('Done.')

    movie_ids = []
    features = []
    for movie_data in tqdm(movies, total=len(movies)):
        movieId = movie_data['movie'].split('/')[-1]
        if movieId not in unique_movies.index:
            continue
        movie_ids.append(movieId)
        # Convert all categorical to binary format and append to list of features
        genres = set(movie_data['genres'].split(','))
        actors = set(movie_data['actors'].split(','))
        directors = set(movie_data['directors'].split(','))
        feats = np.zeros(len(all_genres) + len(all_actors) + len(all_directors))
        with warnings.catch_warnings():
            # hide user warnings about ignored missing values, ignoring these values is the desired behaviour
            warnings.simplefilter("ignore")
            feats[: len(all_genres)] = extract_binary_features(genres, all_genres)
            feats[len(all_genres): len(all_genres) + len(all_actors)] = extract_binary_features(actors, all_actors)
            feats[len(all_genres) + len(all_actors):] = extract_binary_features(directors, all_directors)
            features.append(feats)

    return pd.DataFrame(index=movie_ids, data={'features': features})


def save_set(matrix: pd.DataFrame, name: str):
    matrix.to_csv(name + '.csv', columns=['movieId', 'rating'], mode='w')


if __name__ == '__main__':
    recalculate_metadata = True
    use_genom_tags = True
    save_user_ratings = True
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

    # load movie features from RDF only for movies in movieLens (for which we have ratings)
    if recalculate_metadata:
        print('Loading IMDb data...')
        unique_movies = pd.Series(index=utility_matrix['movieId'].unique().copy())
        imdb_metadata = load_imdb_metadata_features(unique_movies, use_extended=use_audio)
        if use_genom_tags:
            metadata = genome_metadata.join(imdb_metadata, on='movieId', how='inner')
            metadata = pd.DataFrame(index=metadata.index,
                                    data={'features': metadata.apply(
                                        lambda x: np.concatenate([np.array(x.iloc[:-1], dtype=np.float64),
                                                                  np.array(x['features'], dtype=np.float64)],
                                                                 dtype=np.float64),
                                        axis=1)})
        else:
            metadata = imdb_metadata
        print(f'Found {metadata.shape[0]} movies.\nSaving metadata...')
        metadata.to_hdf(item_metadata_file + '.h5', key='metadata', mode='w')
        print('OK!')
    else:
        metadata = pd.read_hdf(item_metadata_file + '.h5', key='metadata')
    # Note to check statistics: metadata['features'].sum(axis=0)
    print(metadata.shape)

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
        print(
            f'Average number of ratings per user (in train set): {user_ratings["rating"].apply(lambda x: len(x)).mean()}')

        if create_user_embeddings_too:
            # create user_embeddings from user ratings once beforehand
            # Note: This takes a very long time
            print('Creating user embeddings...')


            def create_user_embedding(user_ratings: pd.DataFrame, metadata: pd.DataFrame):
                avg_rating = user_ratings['rating'].mean()
                return ((user_ratings['rating'] - avg_rating) * metadata.loc[user_ratings['movieId']][
                    'features'].values).mean()  # TODO: sum or mean?


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
