import matplotlib.pyplot as plt
import csv

"""
File to plot validation and training loss and accuracy over epochs
"""
def get_data_from_csv(csv_path = "../model_data/res.csv"):
    epochs = []
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            split_data = line[0].split(';')
            if split_data[0] == "epoch":
                continue
            epochs.append(int(split_data[0]))
            train_acc.append(float(split_data[1]))
            train_loss.append(float(split_data[2]))
            val_acc.append(float(split_data[3]))
            val_loss.append(float(split_data[4]))
    print(min(train_loss))
    print(min(val_loss))
    print(max(train_acc))
    print(max(val_acc))
    return epochs, train_acc, train_loss, val_acc, val_loss

def plot_loss_data(epochs, train_loss, val_loss):
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_acc_data(epochs, train_acc, val_acc):
    plt.plot(epochs, train_acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_data():
    epochs, train_acc, train_loss, val_acc, val_loss = get_data_from_csv()
    plot_loss_data(epochs, train_loss, val_loss)
    plot_acc_data(epochs, train_acc, val_acc)

plot_data()