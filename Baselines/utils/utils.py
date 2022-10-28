import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_loss, val_loss, loss_name, path, model_name="model"):
    """
    label: training, validation, or test
    loss: np.array
    loss_name: name of the loss fct to plot
    path: where to store the plot
    """
    epochs = np.arange(len(train_loss))
    plt.figure()
    plt.title(loss_name)
    plt.plot(epochs, train_loss, "b-", label="training")
    plt.plot(epochs, val_loss, "g-", label="validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"{path}_{model_name}_{loss_name}.png")
    # plt.show()
    plt.clf()


def prob_to_one_hot(pred):
    """
    Takes a np.array of probabilities and creates the corresponding one-hot encoded np.array where
    we set 1 for the most probable class, all others 0
    """
    arr = np.zeros_like(pred)
    arr[np.arange(len(pred)), pred.argmax(axis=1)] = 1
    return arr
