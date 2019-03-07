# baseline SGD hyper grid search
# search space:
# learning rate: 1e-3, 1e-4, 1e-5
# Dropout rate: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

import sys
sys.path.append("..")
from core import *
import torch
import torchvision
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
import random
import math
import itertools
import pickle

use_cuda = torch.cuda.is_available()

N_Epochs = 600
BatchSize = 128
N_units = 1200

valid_ratio = 1/6

import os.path
dataset_path = os.path.join(os.path.dirname(""), 'mnist_dataset')
if not os.path.exists(dataset_path):
    Download_MNIST = True
else:
    Download_MNIST = False

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: x*255/126])

raw_train_set = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=transform,
    download=Download_MNIST
)

raw_train_size = len(raw_train_set)
valid_size = int(raw_train_size * valid_ratio)
train_size = raw_train_size - valid_size

indices = list(range(raw_train_size))
random.shuffle(indices)
valid_indices = indices[0: valid_size]
train_indices = indices[valid_size: ]

train_set = torch.utils.data.Subset(raw_train_set, train_indices)
valid_set = torch.utils.data.Subset(raw_train_set, valid_indices)

LearningRate_candidates = [1e-3, 1e-4, 1e-5]
DropoutRate_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

train_loader = Data.DataLoader(dataset=train_set, batch_size=BatchSize, shuffle=True)
valid_loader = Data.DataLoader(dataset=valid_set, batch_size=BatchSize, shuffle=True)

N_Train_Batch = train_size / BatchSize

compute_accu = lambda pred, true, digits: round((pred == true).mean() * 100, digits)

hyper_val_error_dict = {}

train_X = Variable(raw_train_set.train_data[train_indices].view(train_size, -1).type(torch.FloatTensor))
train_Y = Variable(raw_train_set.train_labels[train_indices].view(train_size, -1))

if use_cuda:
    train_X, train_Y = train_X.cuda(), train_Y.cuda()

test_X = Variable(raw_train_set.train_data[valid_indices].view(valid_size, -1).type(torch.FloatTensor))
test_Y = Variable(raw_train_set.train_labels[valid_indices].view(valid_size, -1))

if use_cuda:
    test_X, test_Y = test_X.cuda(), test_Y.cuda()

hyper_list = itertools.product(LearningRate_candidates, DropoutRate_candidates)

if __name__ == '__main__':

    for LearningRate, DropoutRate in hyper_list:

      print("*"*50)

      print("Learning rate: {}".format(LearningRate))

      # Initialize network
      net = torch.nn.Sequential(
         torch.nn.Linear(784, N_units),
         torch.nn.ReLU(),
         torch.nn.Dropout(DropoutRate),
         torch.nn.Linear(N_units, N_units),
         torch.nn.ReLU(),
         torch.nn.Dropout(DropoutRate),
         torch.nn.Linear(N_units, 10),
         torch.nn.LogSoftmax(dim=1)
      )

      if use_cuda:
        net = net.cuda()

      loss_fn = torch.nn.NLLLoss(reduction='sum')
      optim = torch.optim.SGD(net.parameters(), lr=LearningRate)

      # Main training loop
      train_accu_lst = []
      test_accu_lst = []

      for i_ep in range(N_Epochs):

          # Training
          net.train()

          loss_nan = False

          for X, Y in train_loader:
              batch_X = Variable(X.view(X.size()[0], -1))
              batch_Y = Variable(Y.view(X.size()[0]))

              if use_cuda:
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()

              y_pred = net(batch_X)

              # Loss and backprop
              loss = loss_fn(y_pred, batch_Y)

              optim.zero_grad()
              loss.backward()
              optim.step()

              if torch.isnan(loss):
                  loss_nan = True

          if loss_nan:
            print("loss nan")
            break

          # Evaluate on training set
          net.eval()

          pred_class = net(train_X).data.cpu().numpy().argmax(axis=1)
          true_class = train_Y.data.cpu().numpy().ravel()

          train_accu = compute_accu(pred_class, true_class, 2)
          print('Epoch', i_ep, '|  Training Accuracy:', train_accu, '%')

          train_accu_lst.append(train_accu)

          # Evaluate on testing set
          pred_class = net(test_X).data.cpu().numpy().argmax(axis=1)
          true_class = test_Y.data.cpu().numpy().ravel()

          test_accu = compute_accu(pred_class, true_class, 2)
          print('Epoch', i_ep, '|  Valid Accuracy:', test_accu, '%')

          test_accu_lst.append(test_accu)

      # to report the final test error, I will use the average of test errors of the last 10 epochs
      report_test_error_mean = round(100-np.average(test_accu_lst[-10:]), 2)
      report_test_error_std = round(np.std(test_accu_lst[-10:]), 2)

      print("Learning Rate: {}".format(LearningRate))
      print("Dropout Rate: {}".format(DropoutRate))
      print("Final report valid error: {} +- {}".format(report_test_error_mean, report_test_error_std))

      hyper_val_error_dict[(LearningRate, DropoutRate)] = (report_test_error_mean, report_test_error_std)

# print results
for key in hyper_val_error_dict.keys():
    print("{}: {}".format(key, hyper_val_error_dict[key]))

result_folder_path = "../../results/Dropout_valid/"
if not os.path.exists(result_folder_path):
    os.mkdir(result_folder_path)
with open(result_folder_path+"valid_error_dict.pkl", 'wb') as f:
    pickle.dump(hyper_val_error_dict, f)
