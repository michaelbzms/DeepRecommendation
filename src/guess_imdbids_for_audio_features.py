import pandas as pd
import numpy as np
from thefuzz import process
import glob

from globals import imdb_path, audio_features_path


def fuzzy_merge(filenames_df, imdb_df, title_col):
    # source: https://stackoverflow.com/questions/13636848/is-it-possible-to-do-fuzzy-match-merge-with-python-pandas
    titles = imdb_df[title_col].tolist()
    filenames_df['fileName'] = filenames_df.index.to_series()
    filenames_df[title_col] = filenames_df['fileName']\
        .apply(lambda file_name: process.extractOne(file_name, titles))\
        .apply(lambda x: x[0])
    imdb_df['movieId'] = imdb_df.index.to_series()
    return pd.merge(filenames_df, imdb_df, on=title_col, how='inner')


def load_imdb(min_votes=12000):
    print('Loading IMDB data')
    all_dfs = []
    for file in ['title.basics.tsv', 'title.ratings.tsv']:
        df = pd.read_csv(imdb_path + file, index_col='tconst',  # usecols=[], dtype={'col': 'UInt32'}
                         sep='\t', encoding='utf-8',
                         keep_default_na=False, na_values=['\\N'])
        all_dfs.append(df)

    # combine all into one big fat DataFrame
    print('concatenating...')
    full_df = pd.concat(all_dfs, axis=1)

    # filter
    full_df = full_df[full_df['titleType'].isin(['tvMovie', 'movie'])]                      # only keep movies
    full_df = full_df[~(full_df['genres'].str.contains('Short', regex=False, na=False))]    # Remove Short movies
    full_df = full_df[(full_df['numVotes'] >= min_votes) & (full_df['primaryTitle'] != 'Yi Yi')]   # keep popular ones
    full_df = full_df.sort_values(by='numVotes', ascending=False)                           # sort by popularity
    movies = pd.DataFrame(full_df[['primaryTitle', 'startYear', 'genres', 'averageRating', 'numVotes']])
    movies['startYear'] = movies['startYear'].fillna(0).astype(np.uint16)
    movies['numVotes'] = movies['numVotes'].fillna(0).astype(np.uint16)
    print('done\nNumber of movies is:', movies.shape[0])

    return movies


features = None
f_list = None
f_names = None
all_dfs = []
for file_path in glob.glob(audio_features_path + '/*.npy'):
    file = file_path.split('\\')[-1]
    # print(file)
    if 'features' in file:      # TODO: Assumes this comes first!!!
        f_names = None
        f_list = None
        features = np.load(file_path)
    elif 'files_list' in file:
        f_list = np.load(file_path)
        for i in range(len(f_list)):
            f_list[i] = f_list[i].split('/')[-1]
    else:
        f_names = np.load(file_path)
    if features is not None and f_list is not None and f_names is not None:
        all_dfs.append(pd.DataFrame(index=f_list, data=features, columns=f_names))
        print(file, 'OK')

print('concatenating...')
audio_df = pd.concat(all_dfs, axis=0)
print(audio_df)
starting_movies = audio_df.shape[0]


# load imdb titles
imdb_df = load_imdb()

# fuzzy merge on the show titles and file names to find most of the IMDbIDs.
# (!) Should be inspected and fixed manually afterwards.
audio_features = fuzzy_merge(audio_df, imdb_df, 'primaryTitle')

final_movies = audio_features.shape[0]
print('Stared with', starting_movies, 'and ended up with', final_movies, 'movies.')

print('Saving findings to csv...')
audio_features.to_csv(audio_features_path + 'imdb_mapped_features.csv', sep=';',
                      index=False, columns=list(f_names) + ['movieId', 'primaryTitle', 'fileName'])
print('Done')
