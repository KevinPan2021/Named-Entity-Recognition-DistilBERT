import matplotlib.pyplot as plt



# plot the loss and acc curves
def plot_training_curves(
        train_acc, train_loss, train_f1,
        valid_acc, valid_loss, valid_f1
    ):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy loss')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.title('Accuracy curves')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy score')
    plt.ylim([0, 100])
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.title('f1 curves')
    plt.xlabel('epochs')
    plt.ylabel('f1 score')
    plt.ylim([0, 100])
    plt.legend()
    plt.show()