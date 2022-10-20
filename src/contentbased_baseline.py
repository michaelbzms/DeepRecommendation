import pandas as pd
import numpy as np
from tqdm import tqdm

from content_providers.fixed_profiles_provider import FixedProfilesProvider
from neural_collaborative_filtering.eval import eval_ranking
from globals import test_set_file


cutoff = 10


def cos_sim(user_profile, item_profiles):
    cos_sim = user_profile @ item_profiles.T
    cos_sim /= np.linalg.norm(user_profile)
    cos_sim /= np.linalg.norm(item_profiles, axis=1)
    return cos_sim


if __name__ == '__main__':
    # test set
    eval_set = pd.read_csv(test_set_file + '.csv')
    eval_set['prediction'] = 0.0

    # Calculate Test NDCG metrics
    cp = FixedProfilesProvider()
    for userId in tqdm(eval_set['userId'], total=len(eval_set['userId'])):
        # get user profile
        user_profile = cp.get_user_profile(userId)
        if np.linalg.norm(user_profile) == 0: continue
        # get item profile
        item_profiles = cp.get_item_profile(eval_set[eval_set['userId'] == userId]['movieId'].values)
        # calculate cos sim
        sim = cos_sim(user_profile, item_profiles)
        # add to pd
        eval_set.loc[eval_set['userId'] == userId, 'prediction'] = sim

    ndcg, adj_ndcg = eval_ranking(eval_set, cutoff=cutoff)
    print(f'Test NDCG@{cutoff}:', ndcg, f'Test adj-NDCG@{cutoff}:', adj_ndcg)

    # Calculate Test+ NDCG metrics
    cp = FixedProfilesProvider(include_val_ratings_to_user_profiles=True)
    for userId in tqdm(eval_set['userId'], total=len(eval_set['userId'])):
        # get user profile
        user_profile = cp.get_user_profile(userId)
        if np.linalg.norm(user_profile) == 0: continue
        # get item profile
        item_profiles = cp.get_item_profile(eval_set[eval_set['userId'] == userId]['movieId'].values)
        # calculate cos sim
        sim = cos_sim(user_profile, item_profiles)
        # add to pd
        eval_set.loc[eval_set['userId'] == userId, 'prediction'] = sim

    ndcg, adj_ndcg = eval_ranking(eval_set, cutoff=cutoff)
    print(f'Test+ NDCG@{cutoff}:', ndcg, f'Test+ adj-NDCG@{cutoff}:', adj_ndcg)
