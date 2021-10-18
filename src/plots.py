import matplotlib.pyplot as plt


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
