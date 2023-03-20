import os

from script.ResultToCSV import CreateHeader

CreateHeader()
os.system('python SingleMachine_MNIST(MLP).py')

os.system('python SingleMachine_MNIST(CNN).py')

os.system('python SingleMachine_CIFAR10(CNN).py')

os.system('python FL_IID_MNIST(MLP).py')

os.system('python FL_IID_MNIST(CNN).py')

os.system('python FL_IID_cifar10(CNN).py')

os.system('python FL_non_IID_MNIST(MLP).py')

os.system('python FL_non_IID_MNIST(CNN).py')

os.system('python FL_non_IID_cifar10(CNN).py')

os.system('python FL_non_IID_Heuristic_MNIST_MLP_.py')

os.system('python FL_non_IID_Heuristic_MNIST_CNN_.py')

os.system('python FL_non_IID_Heuristic_cifar10_CNN_.py')

os.system('python FL_non_IID_ILP_MNIST(MLP).py')

os.system('python FL_non_IID_ILP_MNIST(CNN).py')

os.system('python FL_non_IID_ILP_cifar10(CNN).py')

print("===============Finish===============")

