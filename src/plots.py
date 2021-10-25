import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_train_val_losses(train_losses: [float], val_losses: [float]):
    epochs = list(range(len(train_losses)))
    plt.plot(epochs, train_losses, '-b', label='training loss')
    plt.plot(epochs, val_losses, '-r', label='validation loss')

    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.title('Training and Validation loss during training')
    plt.ylim(ymin=0)

    # save image
    plt.savefig('train_val_loss.png')

    # show
    plt.show()


def plot_fitted_vs_targets(fitted_values: np.array, ground_truth: np.array):
    plt.scatter(ground_truth, fitted_values, marker='_', alpha=0.005)
    plt.plot([0, 5], [0, 5], '--r')
    plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    plt.ylabel('Fitted values')
    plt.xlabel('Ground truth')
    plt.title('Fitted values vs Ground truth')

    # save image
    plt.savefig('fitted_vs_targets.png', dpi=128)

    # show
    plt.show()


def plot_residuals(fitted_values: np.array, ground_truth: np.array):
    plt.style.use('seaborn')
    df = pd.DataFrame(data={'fitted_values': fitted_values, 'ground_truth': ground_truth})
    axes = df['fitted_values'].hist(by=df['ground_truth'], bins=100)
    k = 0.5
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i][j].set_xticks([0, 1, 2, 3, 4, 5])
            axes[i][j].set_yticks([0, 500, 1000])
            # axes[i][j].set_title('Hist of fitted values per ground truth')
            axes[i][j].set_xlabel('Fitted values')
            axes[i][j].axvline(x=k, color='red')
            k += 0.5
            # axes[i][j].set_ylabel('Ground Truth')

    # save image
    plt.savefig('fitted_vs_target_hists.png', dpi=200)

    # show
    plt.show()
