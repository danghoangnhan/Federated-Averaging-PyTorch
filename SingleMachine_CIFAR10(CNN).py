import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from script.ResultToCSV import CreateHeader, CreateResultData, Save_KL_Result, Save_Accuracy_of_each_epoch
from script.getKL import get_KL_value
from model.CIFAR10_CNN import CIFAR10_CNN

batch_size = 1000
num_of_epoch = 5


    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    
    print('-------------------SingleMachine_CIFAR10(CNN)-------------------')
    #####Load and normalize CIFAR10#####
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='../Experiment/data/cifar10', train=True,
                                            download=True, transform=transform)   

    testset = torchvision.datasets.CIFAR10(root='../Experiment/data/cifar10', train=False,
                                           download=True, transform=transform)
    KL_of_each_client, avg_KL = get_KL_value(trainset, 10, 1)

    Save_KL_Result("SingleMachine_CIFAR10(CNN)", KL_of_each_client, avg_KL)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    
      

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


    #####Define a Convolutional Neural Network#####
    net = CIFAR10_CNN()
    #####Define a Loss function and optimizer#####
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #####Train the network#####
    accuracy_of_each_epoch = []
    for epoch in range(num_of_epoch):  # loop over the dataset multiple times
        print(f'[epoch : {epoch + 1}]')
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        eval_accuracy = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {eval_accuracy:.2f} %')
        accuracy_of_each_epoch.append(eval_accuracy)

        
    best_accuracy_of_each_epoch = max(accuracy_of_each_epoch)
    #print("Accuracy list:",accuracy_of_each_epoch)
    #print("Best Accuracy:",best_accuracy_of_each_epoch)
    print('Finished Training')
    PATH = 'cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    #####Test the network on the test data#####
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    net = CIFAR10_CNN()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    #print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                 # for j in range(batch_size)))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total:} %')
    
    total_accuracy = 100 * correct / total
    #print(total_accuracy)
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.2f} %')

    Save_Accuracy_of_each_epoch(1, "SingleMachine_CIFAR10(CNN)", accuracy_of_each_epoch,best_accuracy_of_each_epoch)    

    CreateResultData("SingleMachine_CIFAR10(CNN)", "CIFAR10", "CNN", "", "", num_of_epoch, total_accuracy, "")

        
if __name__ == '__main__':
    main()
