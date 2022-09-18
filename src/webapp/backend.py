import numpy as np
import pandas as pd
import flask
from flask import jsonify, request
import json
import torch

from globals import item_metadata_file
from neural_collaborative_filtering.models.advanced_ncf import AttentionNCF
from neural_collaborative_filtering.util import load_model


app = flask.Flask('movie_recommender_backend')

app.config["DEBUG"] = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# globals
model = None
item_features = None
movie_info = None
ignore_seen = True    # Should we not recommend movies that the user has already seen


@app.route('/movies', methods=['GET'])
def get_movies():
    if movie_info is None: return
    response = jsonify(json.loads(movie_info.to_json(orient='index')))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/test', methods=['GET'])
def get_test():
    return jsonify({'Test': 'This is a test', 'Rating': 5})


@app.route('/recommend', methods=['POST'])
def recommend():
    print('I ran')
    if model is None or item_features is None: return

    # get input json args
    input_json = request.get_json(force=True)
    if app.config["DEBUG"]:
        print('data from client:', input_json)

    # decode JSON input into user ratings
    """ Expects
        user_ratings: str -> [{ imdbId: str,  rating: number }],
        k: number
    """
    user_ratings = {}
    for rating in input_json['user_ratings']:
        user_ratings[rating['imdbID']] = rating['rating']
    user_ratings = pd.Series(index=user_ratings.keys(), data=user_ratings.values(), dtype=float)
    k = input_json['k'] if 'k' in input_json else 10  # default

    if app.config["DEBUG"]:
        print('k =', k)
        print('user_ratings:\n', user_ratings)

    # make recommendations for user
    predictions = recommend_for_user(model, item_features, user_ratings,  k, ignore_seen)

    if app.config["DEBUG"]:
        print('Recommendations:\n', predictions)

    response = jsonify(json.loads(predictions.to_json(orient='records')))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# def create_user_profile(item_features: pd.DataFrame, user_ratings: pd.Series):
#     return ((user_ratings.values - (user_ratings.mean() + 2.5) / 2).reshape(-1, 1) * item_features.loc[user_ratings.index].values).mean(axis=0)
#
#
# def recommend_for_user(model: BasicNCF, item_features: pd.DataFrame, user_ratings: pd.Series,  k, ignore_seen):
#     # determine items to be forwarded
#     items_to_use = item_features.drop(user_ratings.index) if ignore_seen else item_features
#
#     # prepare item input
#     item_input = torch.FloatTensor(items_to_use.values).to(device)
#
#     # build user profile from user ratings (repeat it once for each item)
#     user_features = create_user_profile(item_features, user_ratings)
#     user_input = torch.FloatTensor(user_features.reshape(1, -1).repeat(items_to_use.shape[0], axis=0)).to(device)
#
#     # forward the model
#     with torch.no_grad():
#         y_pred = model(user_input, item_input).cpu().view(-1).numpy()
#
#     # sort the output with their imdb id
#     predictions = pd.DataFrame(data={
#         'imdbID': items_to_use.index,
#         'score': y_pred
#     }).sort_values(by='score', ascending=False).iloc[:k]
#
#     # return sorted imdb_id - predicted score pairs
#     return predictions


def recommend_for_user(model: AttentionNCF, item_features: pd.DataFrame, user_ratings: pd.Series,  k, ignore_seen):
    # candidate item profiles
    items_to_use = item_features.drop(user_ratings.index) if ignore_seen else item_features
    candidate_items = torch.FloatTensor(items_to_use.values).to(device)                 # (B, F)

    # rated items profiles (I, F)
    rated_items_ids = np.sort(np.unique(user_ratings.index))
    rated_items = torch.FloatTensor(item_features.loc[rated_items_ids].values).to(device)

    # user_matrix has shape (B, I) but it's for 1 user so it's just the same row B times
    user_matrix = torch.FloatTensor(np.repeat(np.expand_dims(user_ratings.loc[rated_items_ids].values - ((user_ratings.mean() + 2.5) / 2), axis=0), candidate_items.shape[0], axis=0))

    with torch.no_grad():
        # forward the model
        y_pred, att_weights = model(candidate_items.float().to(device),
                                    rated_items.float().to(device),
                                    user_matrix.float().to(device),
                                    return_attention_weights=True)
        # from torch tensors to numpy vectors
        y_pred = y_pred.view(-1).cpu().numpy()
        att_weights = att_weights.view(-1).cpu().numpy()

    # sort the output with their imdb id
    predictions = pd.DataFrame(data={
        'imdbID': items_to_use.index,
        'score': y_pred
        # TODO: column with list of why
    }).sort_values(by='score', ascending=False).iloc[:k]

    # return sorted imdb_id and predicted score pairs
    return predictions


if __name__ == '__main__':
    # load saved features
    print('Loading movie info...')
    movie_info: pd.DataFrame = pd.read_csv('../../data/movie_info.csv', index_col=0)
    print('Done.')

    # load item input to model
    print('Load item features...')
    metadata: pd.DataFrame = pd.read_hdf(f'../{item_metadata_file}.h5')
    item_features = metadata.loc[movie_info.index]
    print('Done.')

    # load model
    print('Loading model...')
    model_file = '../../models/runs/AttentionNCF_with_features_attNet128_3mil.pt'
    model = load_model(model_file, AttentionNCF)
    model.eval()
    model.to(device)
    print(model)
    print('Done.')

    # run app
    app.run()
