import matplotlib.pyplot as plt
import pandas as pd


def mnist_cnn():
    data1 = pd.read_csv("result/cnn_mnist/1_Accuracy_[MNIST]_CNN C_0.1, E_5, B_10, IID_False.csv", delimiter=',')

    data2 = pd.read_csv("result/cnn_mnist/2_Accuracy_[MNIST]_CNN C_0.1, E_5, B_10, IID_True.csv", delimiter=',')

    data3 = pd.read_csv("result/cnn_mnist/3_Accuracy_[MNIST]_CNN C_0.2, E_5, B_10, IID_False.csv", delimiter=',')

    data4 = pd.read_csv("result/cnn_mnist/4_Accuracy_[MNIST]_CNN C_0.2, E_5, B_10, IID_True.csv", delimiter=',')

    data5 = pd.read_csv("result/cnn_mnist/5_Accuracy_[MNIST]_CNN C_0.5, E_5, B_10, IID_False.csv", delimiter=',')

    data6 = pd.read_csv("result/cnn_mnist/6_Accuracy_[MNIST]_CNN C_0.5, E_5, B_10, IID_True.csv", delimiter=',')

    data7 = pd.read_csv("result/cnn_mnist/7_Accuracy_[MNIST]_CNN C_1.0, E_5, B_10, IID_False.csv", delimiter=',')

    data8 = pd.read_csv("result/cnn_mnist/8_Accuracy_[MNIST]_CNN C_1.0, E_5, B_10, IID_True.csv", delimiter=',')

    plt.plot(data1["Step"][:300], data1["Value"][:300], color='blue', label='C=0.1, E=5, B=10, NonIID')

    plt.plot(data2["Step"][:300], data2["Value"][:300], color='red', label='C=0.1, E=5, B=10, IID')

    plt.plot(data3["Step"][:300], data3["Value"][:300], color='green', label='C=0.2, E=5, B=10, NonIID')

    plt.plot(data4["Step"][:300], data4["Value"][:300], color='black', label='C=0.2, E=5, B=10, IID')

    plt.plot(data5["Step"][:300], data5["Value"][:300], color='pink', label='C=0.5, E=5, B=10, NonIID')

    plt.plot(data6["Step"][:300], data6["Value"][:300], color='yellow', label='C=0.5, E=5, B=10, IID')

    plt.plot(data7["Step"][:300], data7["Value"][:300], color='grey', label='C=1.0, E=5, B=10, NonIID')

    plt.plot(data8["Step"][:300], data8["Value"][:300], color='purple', label='C=1.0, E=5, B=10, IID')

    # plt.title("experiment on MINIST data set")  # title
    plt.ylabel("Accuracy")  # y label
    plt.xlabel("Comunication Rounds")  # x label
    plt.legend()
    plt.savefig('MINIST_CNN.png')


def mnist_twonn():
    data1 = pd.read_csv("result/twonn_mnist/12_Accuracy_[MNIST]_TwoNN C_0.2, E_5, B_10, IID_True.csv", delimiter=',')

    data2 = pd.read_csv("result/twonn_mnist/13_Accuracy_[MNIST]_TwoNN C_0.5, E_5, B_10, IID_False.csv", delimiter=',')

    data3 = pd.read_csv("result/twonn_mnist/15_Accuracy_[MNIST]_TwoNN C_1.0, E_5, B_10, IID_False.csv", delimiter=',')

    plt.plot(data1["Step"][:300], data1["Value"][:300], color='blue', label='C=0.1, E=5, B=10, IID')

    plt.plot(data2["Step"][:300], data2["Value"][:300], color='red', label='C=0.1, E=5, B=10, NonIID')

    plt.plot(data3["Step"][:300], data3["Value"][:300], color='green', label='C=0.2, E=5, B=10, NonIID')

    # plt.title("experiment on MINIST data set")  # title
    plt.ylabel("Accuracy")  # y label
    plt.xlabel("Comunication Rounds")  # x label
    plt.legend()
    plt.savefig('MINIST_TwoNN.png')
def cifar10_cnn():
    data1 = pd.read_csv("result/cnn_cifar10/17_Accuracy_[CIFAR10]_CNN2 C_0.1, E_5, B_10, IID_False.csv", delimiter=',')

    data2 = pd.read_csv("result/cnn_cifar10/19_Accuracy_[CIFAR10]_CNN2 C_0.2, E_5, B_10, IID_False.csv", delimiter=',')

    data3 = pd.read_csv("result/cnn_cifar10/21_Accuracy_[CIFAR10]_CNN2 C_0.5, E_5, B_10, IID_False.csv", delimiter=',')

    data4 = pd.read_csv("result/cnn_cifar10/23_Accuracy_[CIFAR10]_CNN2 C_1.0, E_5, B_10, IID_False.csv", delimiter=',')


    plt.plot(data1["Step"][:300], data1["Value"][:300], color='blue', label='C=0.1, E=5, B=10, NonIID')

    plt.plot(data2["Step"][:300], data2["Value"][:300], color='red', label='C=0.2, E=5, B=10, NonIID')

    plt.plot(data3["Step"][:300], data3["Value"][:300], color='green', label='C=0.3, E=5, B=10, NonIID')

    plt.plot(data4["Step"][:300], data4["Value"][:300], color='green', label='C=0.5, E=5, B=10, NonIID')


    # plt.title("experiment on MINIST data set")  # title
    plt.ylabel("Accuracy")  # y label
    plt.xlabel("Comunication Rounds")  # x label
    plt.legend()
    plt.savefig('CIFAR10_CNN.png')


if __name__ == '__main__':
    # mnist_cnn()
    # mnist_twonn()
    cifar10_cnn()