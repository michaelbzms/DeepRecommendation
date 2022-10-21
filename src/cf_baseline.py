import pandas as pd
from matrix_factorization import KernelMF

from globals import train_set_file, test_set_file, val_set_file
from sklearn.metrics import mean_squared_error

from neural_collaborative_filtering.eval import eval_ranking


def read_data(file):
    df = pd.read_csv(file + '.csv', skiprows=1, names=['user_id', 'item_id', 'rating'])
    df['item_id'] = df['item_id'].apply(lambda x: int(x[2:]))   # convert to int ids
    return df


if __name__ == '__main__':
    # Source: https://pypi.org/project/matrix-factorization/

    # read data
    train_df = read_data(train_set_file)
    val_df = read_data(val_set_file)
    test_df = read_data(test_set_file)

    num_epochs = [64, 80, 100, 128]
    lrs = [0.001, 0.0005]
    num_factors = [64, 128, 256]
    regs = [0.01, 0.005]
    best_val_mse = 2 ** 10
    best_model = None

    # grid search best hyperparams according to val mse
    for n_factors in num_factors:
        for lr in lrs:
            for reg in regs:
                for n_epochs in num_epochs:
                    # Matrix Factorization
                    matrix_fact = KernelMF(n_epochs=n_epochs, n_factors=n_factors, lr=lr, reg=reg, verbose=2)

                    # fit train data
                    matrix_fact.fit(train_df[['user_id', 'item_id']], train_df['rating'])

                    # predict val set
                    pred = matrix_fact.predict(val_df[['user_id', 'item_id']])
                    val_mse = mean_squared_error(val_df['rating'], pred, squared=True)
                    print(f'Factors: {n_factors}, Learning rate: {lr}, reg: {reg}, Epochs: {n_epochs} --> Val MSE: {val_mse: .4f}')
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_model = matrix_fact

    # predict test set
    pred = best_model.predict(test_df[['user_id', 'item_id']])
    test_mse = mean_squared_error(test_df['rating'], pred, squared=True)
    print(f'\nTest MSE: {test_mse: .4f} - Val MSE: {best_val_mse: .4f}')

    ndcg_df = pd.DataFrame(data={
        'userId': test_df['user_id'],
        'movieId': test_df['item_id'],
        'rating': test_df['rating'],
        'prediction': pred
    })
    test_ndcg, test_ndcg_adj = eval_ranking(ndcg_df, cutoff=10)
    print(f'\nTest NDCG@10: {test_ndcg: .4f} - Test adj-NDCG@10: {test_ndcg_adj: .4f}')
