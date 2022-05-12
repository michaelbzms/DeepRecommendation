import pandas as pd
import numpy as np
from tqdm import tqdm

from content_providers.fixed_profiles_provider import FixedProfilesProvider
from globals import val_set_file
from neural_collaborative_filtering.eval import eval_ranking


def cos_sim(user_profile, item_profiles):
    cos_sim = user_profile @ item_profiles.T
    cos_sim /= np.linalg.norm(user_profile)
    cos_sim /= np.linalg.norm(item_profiles, axis=1)
    return cos_sim


if __name__ == '__main__':
    cp = FixedProfilesProvider()
    eval_set = pd.read_csv(val_set_file + '.csv')
    eval_set['prediction'] = 0.0
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

    cutoff = 10
    ndcg, adj_ndcg = eval_ranking(eval_set, cutoff=cutoff)
    print(f'NDCG@{cutoff}:', ndcg, f'adj-NDCG@{cutoff}:', adj_ndcg)
