import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from rdflib import Graph, Namespace
from tqdm import tqdm
import warnings

from globals import movielens_path, rdf_path, save_movie_metadata_to


def load_user_ratings(movielens_data_folder, limit=None) -> pd.DataFrame:
    # load movielens user reviews data
    user_ratings = pd.read_csv(movielens_data_folder + 'ratings.csv',
                               index_col='userId',
                               usecols=['userId', 'movieId', 'rating', 'timestamp'],
                               dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
    if limit is not None:
        user_ratings = user_ratings[:limit]
    # link movieIds with imdbIds
    links = pd.read_csv(movielens_data_folder + 'links.csv',
                        index_col='movieId',
                        usecols=['movieId', 'imdbId'],
                        dtype={'movieId': np.int32, 'imdbId': 'string'})
    user_ratings['movieId'] = 'tt' + user_ratings['movieId'].map(links['imdbId'])
    return user_ratings


def extract_binary_features(actual_values: set, ordered_possible_values: list) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform([actual_values])
    return binary_format


def load_movie_metadata_features(unique_movies: pd.Series):
    # NAMESPACES
    ns_movies = Namespace('https://www.imdb.com/title/')
    ns_genres = Namespace('https://www.imdb.com/search/title/?genres=')
    ns_principals = Namespace('https://www.imdb.com/name/')
    ns_predicates = Namespace('http://example.org/props/')

    print('Loading rdf...')
    rdf = Graph().parse(rdf_path + 'movies_pruned_actors.nt', format='nt')
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


if __name__ == '__main__':
    # load user ratings (sparse representation of a utility matrix)
    print('Loading movieLens data...')
    utility_matrix = load_user_ratings(movielens_path)
    print(utility_matrix)

    # load movie features from RDF only for movies in movieLens (for which we have ratings)
    print('Loading IMDb data...')
    unique_movies = pd.Series(index=utility_matrix['movieId'].unique().copy())
    metadata = load_movie_metadata_features(unique_movies)
    # TODO: save? Can't use to_csv() because of encapsulated numpy array.
    # Note to check statistics: metadata['features'].sum(axis=0)

    # Note: there can still be movies in ratings for which we have no features -> TODO: ignore them

    # train-val-test split (global temporal splitting)
    print('Calculating splits...')
    global_val_split = utility_matrix['timestamp'].groupby('userId').quantile(0.8).mean()
    global_test_split = utility_matrix['timestamp'].groupby('userId').quantile(0.9).mean()
    # print(f'Global split time percentage: {100.0 * (global_val_split - utility_matrix["timestamp"].min()) / (utility_matrix["timestamp"].max() - utility_matrix["timestamp"].min()):.2f}%')
    # print(f'Val set percentage of held out set: {100.0 * (global_test_split - global_val_split) / (utility_matrix["timestamp"].max() - global_val_split):.2f}%')

