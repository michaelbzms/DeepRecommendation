import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from tqdm import tqdm

from recommendation.utility_matrix import UtilityMatrix


def create_onehot_graph(all_users: np.array, all_items: np.array, graph_edges, user_ratings):
    """ Create the graph from user and item nodes and use given edges with weights between them """
    print('Creating graph...')
    # First sorted items then sorted users as nodes with combined one-hot vector representations
    x = torch.eye(len(all_items) + len(all_users))

    all_items_index = {i: ind for ind, i in enumerate(all_items)}
    all_users_index = {u: ind + len(all_items) for ind, u in enumerate(all_users)}       # IMPORTANT: add num items to user index!!!

    # find edges
    edge_index = [[all_users_index[u] for u in graph_edges['userId']],
                  [all_items_index[i] for i in graph_edges['movieId']]]
    # append backward edges too
    edge_index[0] += edge_index[1]
    edge_index[1] += edge_index[0][:graph_edges.shape[0]]

    # TODO: This is dumb
    # edge_attr = [[rating] for rating in graph_edges['rating']] * 2

    # Note: use rating - avg user rating instead of just the rating. Sign is meaningful this way
    # Note: Negative weights give nan values. Why??? -> because of sqrt(node_degree). Setting normalize=False fixes it
    edge_attr = [[(edge['rating'] - user_ratings.loc[int(edge['userId'])]['meanRating'])]
                 for _, edge in tqdm(graph_edges.iterrows(), desc='Loading graph edges...', total=len(graph_edges))] * 2

    # edge_index = [[all_users_index[edge['userId']] for _, edge in graph_edges.iterrows()
    #                if edge['rating'] > user_ratings.loc[int(edge['userId'])]['meanRating']],
    #               [all_items_index[edge['movieId']] for _, edge in graph_edges.iterrows()
    #                if edge['rating'] > user_ratings.loc[int(edge['userId'])]['meanRating']]]
    # # append backward edges
    # start_len = len(edge_index[0])
    # edge_index[0] += edge_index[1]
    # edge_index[1] += edge_index[0][:start_len]
    #
    # # Add only edges that are higher that the user's average ratings
    # edge_attr = [[-edge['rating']]
    #              for _, edge in tqdm(graph_edges.iterrows(), desc='Loading graph edges...', total=len(graph_edges))
    #              if edge['rating'] > user_ratings.loc[int(edge['userId'])]['meanRating']] * 2

    known_graph = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        num_items=len(all_items),
        edge_dim=1
    )
    print(known_graph)

    # remove duplicates
    print('Removing duplicate edges...')
    known_graph.edge_index, known_graph.edge_attr = coalesce(known_graph.edge_index, known_graph.edge_attr, reduce='mean')
    print(known_graph)
    print('done.')

    return known_graph, all_users_index, all_items_index


def create_onehot_graph_from_utility_matrix(utility_matrix: UtilityMatrix, all_items, all_users, genres=None):
    print('Creating graph...')

    # mark unique index to all
    all_items_index = {i: ind for ind, i in enumerate(all_items)}
    all_users_index = {u: ind + len(all_items) for ind, u in enumerate(all_users)}  # IMPORTANT: add num items to user index!!!

    edge_index = [[], []]
    edge_attr = []
    edge_dim = 3   # TODO
    for _, (userId, itemId, rating) in tqdm(utility_matrix.sparse_matrix.iterrows(), desc='Loading graph edges...', total=len(utility_matrix.sparse_matrix)):
        # add edge user ----> item with weight: rating - avg_user_rating
        edge_index[0].append(all_users_index[userId])
        edge_index[1].append(all_items_index[itemId])
        edge_attr.append([rating, rating - utility_matrix.get_user_mean_rating(userId), 0])
        # add edge item ----> user with weight: rating - avg_item_rating
        edge_index[0].append(all_items_index[itemId])
        edge_index[1].append(all_users_index[userId])
        edge_attr.append([rating, rating - utility_matrix.get_item_mean_rating(itemId), 0])

    if genres is not None:
        all_genres = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Biography', 'Music'
            # Rare (for now) categories: 'History', 'Family', 'Sport'
        ]
        all_genres_index = {n: ind + len(all_items) + len(all_users) for ind, n in enumerate(all_genres)}
        for itemId in tqdm(all_items, desc='Adding genre nodes...'):
            try:
                gs = genres.loc[itemId]['genres'].split(',')
                for g in gs:
                    if g in all_genres:
                        # add item ---> genre edge
                        edge_index[0].append(all_items_index[itemId])
                        edge_index[1].append(all_genres_index[g])
                        edge_attr.append([0] * (edge_dim - 1) + [1])   # default attributes
                        # add genre ---> item edge
                        edge_index[0].append(all_genres_index[g])
                        edge_index[1].append(all_items_index[itemId])
                        edge_attr.append([0] * (edge_dim - 1) + [1])
            except KeyError:
                print('Warning: Could not find genres for an item!')

        x = torch.eye(len(all_items) + len(all_users) + len(all_genres))
    else:
        x = torch.eye(len(all_items) + len(all_users))

    known_graph = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        num_items=len(all_items),
        edge_dim=edge_dim
    )
    print(known_graph)
    print('done.')

    return known_graph, all_users_index, all_items_index
