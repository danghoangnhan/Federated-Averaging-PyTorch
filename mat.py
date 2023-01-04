import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv("result/cnn_mnist/1_Accuracy_[MNIST]_CNN C_0.1, E_5, B_10, IID_False.csv", delimiter=',')

data2 = pd.read_csv("result/cnn_mnist/2_Accuracy_[MNIST]_CNN C_0.1, E_5, B_10, IID_True.csv", delimiter=',')

data3 = pd.read_csv("result/cnn_mnist/3_Accuracy_[MNIST]_CNN C_0.2, E_5, B_10, IID_False.csv", delimiter=',')

data4 = pd.read_csv("result/cnn_mnist/4_Accuracy_[MNIST]_CNN C_0.2, E_5, B_10, IID_True.csv", delimiter=',')

data5 = pd.read_csv("result/cnn_mnist/5_Accuracy_[MNIST]_CNN C_0.5, E_5, B_10, IID_False.csv", delimiter=',')

data6 = pd.read_csv("result/cnn_mnist/6_Accuracy_[MNIST]_CNN C_0.5, E_5, B_10, IID_True.csv", delimiter=',')

data7 = pd.read_csv("result/cnn_mnist/7_Accuracy_[MNIST]_CNN C_1.0, E_5, B_10, IID_False.csv", delimiter=',')

data8 = pd.read_csv("result/cnn_mnist/8_Accuracy_[MNIST]_CNN C_1.0, E_5, B_10, IID_True.csv", delimiter=',')

plt.plot(data1["Step"][:400], data1["Value"][:400], color='blue', label='C=0.1, E=5, B=10, NonIID')

plt.plot(data2["Step"][:400], data2["Value"][:400], color='red', label='C=0.1, E=5, B=10, IID')

plt.plot(data3["Step"][:400], data3["Value"][:400], color='green', label='C=0.2, E=5, B=10, NonIID')

plt.plot(data4["Step"][:400], data4["Value"][:400], color='black', label='C=0.2, E=5, B=10, IID')

plt.plot(data5["Step"][:400], data5["Value"][:400], color='pink', label='C=0.5, E=5, B=10, NonIID')

plt.plot(data6["Step"][:400], data6["Value"][:400], color='yellow', label='C=1.0, E=5, B=10, NonIID')

plt.plot(data7["Step"][:400], data7["Value"][:400], color='grey', label='C=1.0, E=5, B=10, NonIID')

plt.plot(data8["Step"][:400], data8["Value"][:400], color='purple', label='C=1.0, E=5, B=10, NonIID')

# plt.title("experiment on MINIST data set")  # title
plt.ylabel("Accuracy")  # y label
plt.xlabel("Comunication Rounds")  # x label
plt.legend()

plt.show()
