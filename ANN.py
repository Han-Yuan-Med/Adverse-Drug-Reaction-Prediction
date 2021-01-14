from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import torch.utils.data as tud
import numpy as np
import pandas as pd
from pandas import concat
from sklearn import preprocessing
from sklearn.preprocessing import scale
import random
from pandas import DataFrame

#batch_size = 1024

with open("datsetEB.csv") as f:
    datasetEB = pd.read_csv(f)
f.close()

with open("test_positive.csv") as f:
    test_positive = pd.read_csv(f)
f.close()

with open("test_negative.csv") as f:
    test_negative = pd.read_csv(f)
f.close()

datasetEB.head()
test_negative.head()

train_dataset = pd.DataFrame(datasetEB).values
test_dataset_positive = pd.DataFrame(test_positive).values
test_dataset_negative = pd.DataFrame(test_negative).values

x = train_dataset[:, range(188)]
x_positive = test_dataset_positive[:, range(188)]
x_negative = test_dataset_negative[:, range(188)]

min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)
x_positive = min_max_scaler.fit_transform(x_positive)
x_negative = min_max_scaler.fit_transform(x_negative)

y = train_dataset[:, 188]
y1 = test_dataset_positive[:, 188]
y2 = test_dataset_negative[:, 188]

class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(188, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2)

    def forward(self, m):
        m = F.relu(self.fc1(m))
        m = F.relu(self.fc2(m))
        m = F.softmax(self.fc3(m), dim=1)
        return m

#weights = [1, 1]
weights = [0.21875, 0.78125]
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))

def testpositive():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_positive:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader_positive.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_positive.dataset),
        100. * correct / len(test_loader_positive.dataset)))

    return output

def testnegative():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_negative:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader_negative.dataset)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_negative.dataset),
        100. * correct / len(test_loader_negative.dataset)))

    return output

prob_positive = np.zeros((12, 400))
prob_negative = np.zeros((9211, 400))

for i in range(1, 201):
    seed = i
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    x_minmax_list = list(x_minmax)
    y_list = list(y)
    x_onetime = x_minmax_list[(128*(i-1)):(128*(i-1)+127)]
    x_onetime = np.array(x_onetime)
    y_onetime = y_list[(128*(i-1)):(128*(i-1)+127)]
    y_onetime = np.array(y_onetime)
    x_train = torch.FloatTensor(x_onetime)
    x_test_positive = torch.FloatTensor(x_positive)
    x_test_negative = torch.FloatTensor(x_negative)
    y_train = torch.LongTensor(y_onetime)
    y_test_positive = torch.LongTensor(y1)
    y_test_negative = torch.LongTensor(y2)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset_positive = torch.utils.data.TensorDataset(x_test_positive, y_test_positive)
    test_dataset_negative = torch.utils.data.TensorDataset(x_test_negative, y_test_negative)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader_positive = torch.utils.data.DataLoader(dataset=test_dataset_positive, batch_size=12, shuffle=False)
    test_loader_negative = torch.utils.data.DataLoader(dataset=test_dataset_negative, batch_size=9211, shuffle=False)

    model = MLP()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, 500):
        train(epoch)

    a = testpositive()
    a
    b = testnegative()
    b
    c = a.numpy()
    d = b.numpy()
    prob_positive[:, [2*(i-1), (2*i-1)]] = c
    prob_negative[:, [2 * (i - 1), (2 * i - 1)]] = d

np.savetxt('prob_positive.csv', prob_positive, delimiter=',')
np.savetxt('prob_negative.csv', prob_negative, delimiter=',')


