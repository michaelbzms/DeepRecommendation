import matplotlib.pyplot as plt
import numpy as np


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
    plt.scatter(ground_truth, fitted_values, marker='_', alpha=0.01)
    plt.plot([0, 5], [0, 5], '--r')
    plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    plt.ylabel('Fitted values')
    plt.xlabel('Ground truth')
    plt.title('Fitted values vs Ground truth')

    # save image
    plt.savefig('fitted_vs_targets.png', dpi=150)

    # show
    plt.show()


def plot_residuals(residuals: np.array):
    # TODO: group by ground truth, more bins

    plt.hist(residuals)
    plt.title('Residuals')

    # save image
    plt.savefig('residuals.png', dpi=100)

    # show
    plt.show()
