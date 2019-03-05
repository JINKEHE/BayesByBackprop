# the baseline model: SGD without Dropout or L2 Regularization

import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable
import os.path
import numpy as np

use_cuda = torch.cuda.is_available()

N_Epochs = 600
LearningRate = 1e-3
BatchSize = 128
Download_MNIST = True   # download the dataset if you don't already have it

dataset_path = os.path.join(os.path.dirname(""), 'mnist')

train_set = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)

train_size = train_set.train_data.size()[0]
N_Batch = train_size / BatchSize

train_loader = Data.DataLoader(dataset=train_set, batch_size=BatchSize, shuffle=True)

test_set = torchvision.datasets.MNIST(
    root=dataset_path,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)

test_size = test_set.test_data.size()[0]
compute_accu = lambda pred, true, digits: round((pred == true).mean() * 100, digits)

if __name__ == '__main__':

    # Initialize network
    net =  torch.nn.Sequential(
              torch.nn.Linear(784, 800),
              torch.nn.ReLU(),
              torch.nn.Linear(800, 800),
              torch.nn.ReLU(),
              torch.nn.Linear(800, 10),
              torch.nn.LogSoftmax())

    if use_cuda:
      net = net.cuda()

    optim = torch.optim.SGD(net.parameters(), lr=LearningRate)
    loss_fn = torch.nn.NLLLoss(reduction='sum')

    # Main training loop
    train_accu_lst = []
    test_accu_lst = []

    for i_ep in range(N_Epochs):

        # Training
        net.train()
        for X, Y in train_loader:
            batch_X = Variable(X.view(X.size()[0], -1))
            batch_Y = Variable(Y.view(X.size()[0]))

            if use_cuda:
              batch_X = batch_X.cuda()
              batch_Y = batch_Y.cuda()

            y_pred = net(batch_X)

            # Loss and backprop
            loss = loss_fn(y_pred, batch_Y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Evaluate on training set
        net.eval()
        train_X = Variable(train_set.train_data.view(train_size, -1).type(torch.FloatTensor))
        train_Y = Variable(train_set.train_labels.view(train_size, -1))

        if use_cuda:
          train_X = train_X.cuda()
          train_Y = train_Y.cuda()

        pred_class = net(train_X).cpu().data.numpy().argmax(axis=1)
        true_class = train_Y.cpu().data.numpy().ravel()

        train_accu = compute_accu(pred_class, true_class, 2)
        print('Epoch', i_ep, '|  Training Accuracy:', train_accu, '%')

        train_accu_lst.append(train_accu)

        # Evaluate on test set
        test_X = Variable(test_set.test_data.view(test_size, -1).type(torch.FloatTensor))
        test_Y = Variable(test_set.test_labels.view(test_size, -1))

        if use_cuda:
          test_X = test_X.cuda()
          test_Y = test_Y.cuda()

        pred_class = net(test_X).cpu().data.numpy().argmax(axis=1)
        true_class = test_Y.cpu().data.numpy().ravel()

        test_accu = compute_accu(pred_class, true_class, 2)
        print('Epoch', i_ep, '|  Test Accuracy:', test_accu, '%', '| Error Rate:', round(100.00-test_accu, 2), '%')

        test_accu_lst.append(test_accu)
    report_test_accuracy_mean = np.mean(test_accu_lst[-10:])
    report_test_accuracy_std = np.std(test_accu_lst[-10:])
    report_test_error_mean = round(100-report_test_accuracy_mean, 2)
    print("test accuracy mean: {}".format(report_test_accuracy_mean))
    print("test accuracy std: {}".format(report_test_accuracy_std))
    print("test error mean: {}".format(report_test_error_mean))
