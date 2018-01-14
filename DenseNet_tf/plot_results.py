import matplotlib.pylab as plt
import json
import numpy as np


def plot_cifar10(save=True):

    with open("./logs/running.json", "r") as f:
        d = json.load(f)

    train_accuracy = 100 * np.array(d["train_acc"])
    test_accuracy = 100 * np.array(d["test_acc"])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Accuracy')
    ax1.plot(train_accuracy, color="tomato", linewidth=2, label='train_acc')
    ax1.plot(test_accuracy, color="steelblue", linewidth=2, label='test_acc')
    ax1.legend(loc=0)

    train_loss = np.array(d["train_loss"])
    test_loss = np.array(d["test_loss"])


    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(train_loss, '--', color="tomato", linewidth=2, label='train_loss')
    ax2.plot(test_loss, '--', color="steelblue", linewidth=2, label='test_loss')
    ax2.legend(loc=1)

    ax1.grid(True)

    if save:
        fig.savefig('./figures/plot_training.svg')

    fig_lr = plt.figure()
    ax_lr = fig_lr.add_subplot(111)
    ax_lr.set_ylabel('learning_rate')
    learning_rate = np.array(d["learning_rate"])
    ax_lr.set_ylim([0, learning_rate[0]])
    ax_lr.plot(learning_rate, color="tomato", linewidth=2, label='learning_rate')
    ax_lr.grid(True)

    plt.show()
    plt.clf()
    plt.close()



if __name__ == '__main__':
    plot_cifar10()
