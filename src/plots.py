import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from globals import plots_path


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
    plt.savefig(plots_path +'train_val_loss.png')

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
    plt.savefig(plots_path +'fitted_vs_targets.png', dpi=128)

    # show
    plt.show()


def plot_residuals(fitted_values: np.array, ground_truth: np.array):
    plt.style.use('seaborn')
    df = pd.DataFrame(data={'fitted_values': fitted_values, 'ground_truth': ground_truth})
    axes = df['fitted_values'].hist(by=df['ground_truth'], bins=100, stacked=True)
    k = 0.5
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i][j].set_xticks([0, 1, 2, 3, 4, 5])
            axes[i][j].set_yticks([0, 250, 500])
            # axes[i][j].set_title('Hist of fitted values per ground truth')
            axes[i][j].set_xlabel('Fitted values')
            axes[i][j].axvline(x=k, color='red')
            k += 0.5
            # axes[i][j].set_ylabel('Ground Truth')

    # save image
    plt.savefig(plots_path +'fitted_vs_target_hists.png', dpi=200)

    # show
    plt.show()


def plot_stacked_residuals(fitted_values, ground_truth, normalize=True):
    # plt.style.use('seaborn')
    df = pd.DataFrame(index=ground_truth, data={'fitted_values': fitted_values})
    if normalize:
        num_bins = 128
        _, bins = np.histogram(df['fitted_values'].values, bins=num_bins)
        pre_height = np.zeros(num_bins)
        for k in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
            hist, _ = np.histogram(df.loc[k]['fitted_values'].values, bins=bins)  # pass actual bin points
            height = np.array(hist.astype(np.float32) / hist.sum())
            plt.bar((bins[:-1] + bins[1:]) / 2, height, width=(bins[1] - bins[0]), bottom=pre_height, label=str(k))
            pre_height += height
        plt.legend(title='Ground truth')
    else:
        plt.hist([df.loc[k]['fitted_values'] for k in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]],
                 bins=128, stacked=True)
        plt.legend(title='Ground truth', labels=[str(num) for num in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]])
    plt.title(('Normalized (per target) histogram' if normalize else 'Histogram') + ' of predicted ratings per ground truth')
    plt.xlabel('Predicted ratings')
    plt.ylabel('Frequency')

    # save image
    plt.savefig(plots_path +'fitted_hist.png', dpi=150)

    # show
    plt.show()


def visualize_attention(weights: np.array, user_matrix: np.array, candidate_names, rated_names):
    B, I = len(candidate_names), len(rated_names)

    fig, ax = plt.subplots(figsize=(19, 9))
    # fig.set_size_inches(18.0, 9.0)
    # ax.imshow(np.array(weights), cmap='hot', interpolation='nearest')
    ax = sns.heatmap(np.array(weights), robust=True, annot=False, linewidths=0.5, square=True, cbar=True)
    print(weights)

    # We want to show all ticks...
    ax.set_yticks(np.arange(B))
    ax.set_xticks(np.arange(I))
    # ... and label them with the respective list entries
    ax.set_yticklabels([name[0] for name in candidate_names], fontsize=8)
    ax.set_xticklabels([name[0] for name in rated_names.values], fontsize=8)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(B):
    #     for j in range(I):
    #         text = ax.text(j, i, float(weights[i, j]), ha="center", va="center")

    ax.set_title("Attention weights visualization")
    fig.tight_layout()
    plt.show()
