# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from time import *
from tqdm import tqdm


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.linear1 = torch.nn.Linear(20 * 5 * 5, 50)
        self.drop1 = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(50, 10)


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        in_linear1 = out_conv2.view(out_conv2.size(0), -1)
        out_linear1 = self.linear1(in_linear1)
        out_drop = self.drop1(out_linear1)
        out = self.linear2(out_drop)
        return out


data_transform = transforms.ToTensor()

train_data = MNIST(root='./data', train=True, transform=data_transform, download=True)
test_data = MNIST(root='./data', train=False, transform=data_transform, download=True)

print('train_data number: ', len(train_data))
print('Test_data number: ', len(test_data))

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
print(device)

batch_sizes = [32, 128, 512]
results = []
for batch_size in batch_sizes:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    my_model = Net()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    loss_func = torch.nn.CrossEntropyLoss()

    if use_gpu:
        my_model = my_model.to(device)
        loss_func = loss_func.to(device)

    print('batch_size: ', batch_size, ' begin!')


    def train(EPOCH):
        loss_list = []
        sum = 0.0
        sum_time = 0.0
        for epoch in tqdm(range(EPOCH), desc='Processing'):
            for batch, data in enumerate(train_loader):
                begin_time = time()
                x, y = data
                if use_gpu:
                    x, y = x.to(device), y.to(device)
                x, y = Variable(x), Variable(y)
                y_pred = my_model(x)
                loss = loss_func(y_pred, y)
                sum += float(loss)
                loss_list.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                end_time = time()
                sum_time += end_time - begin_time
            if epoch % 9 == 0 and epoch != 0:
                print('Epoch%d~%d ' % (epoch - 9, epoch), ' Avg loss: ', sum / 10,
                      'avg spend time: %f' % (sum_time / 10))
                sum = 0.0
                sum_time = 0.0
        return loss_list


    def test():
        y_pred = np.array([], dtype=int)
        y_true = np.array([], dtype=int)
        for batch_i, data in tqdm(enumerate(test_loader), desc='Processing'):
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            outputs = my_model(inputs)
            # loss = loss_func(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.data.view_as(predicted).cpu().numpy()
            predicted = predicted.cpu().numpy()
            y_pred = np.append(y_pred, predicted)
            y_true = np.append(y_true, labels)
            # compare predictions to true label
            # print(predicted)
            # print(labels)
            # correct += np.squeeze(predicted.eq(labels.data.view_as(predicted))).sum()
        # acc = correct / len(test_data)
        # print('accurary: ',acc)
        return (y_pred, y_true)


    train(100)
    result = test()
    print(classification_report(result[1], result[0]))
    results.append(result)
# print('results of batch_size in [32, 128, 512]: ', results)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
