from core import *
import torch
import torchvision
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
import random
import math
import itertools

use_cuda = torch.cuda.is_available()

# hypers that do not need to be tuned
N_Epochs = 100 # in actual training, we use 600
BatchSize = 128

valid_ratio = 1/6

Download_MNIST = True

import os.path
dataset_path = os.path.join(os.path.dirname(""), 'mnist')

raw_train_set = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=torchvision.transforms.ToTensor(),
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

# TODO: cannot specify the hyper for priors outside the network

# need N_epochs * 4 * 2 * 3 * 3 * 3 = 100 * 4 * 54= 21600 epochs = 648000 seconds = 180 hours
N_Samples_Testing_candidates = [1,2,5,10]
LearningRate_candidates = [1e-3, 1e-4, 1e-5]
mixture_PI_candidates = [0.25, 0.5, 0.75]
mixture_sigma1_candidates = [math.exp(-0), math.exp(-1), math.exp(-2)]
mixture_sigma2_candidates = [math.exp(-6), math.exp(-7), math.exp(-8)]

train_loader = Data.DataLoader(dataset=train_set, batch_size=BatchSize, shuffle=True)
valid_loader = Data.DataLoader(dataset=valid_set, batch_size=BatchSize, shuffle=True)

N_Train_Batch = train_size / BatchSize

compute_accu = lambda pred, true, digits: round((pred == true).mean() * 100, digits)

hyper_val_error_dict = {}

if __name__ == '__main__':

    # could may have more
    hyper_list = itertools.product(LearningRate_candidates, N_Samples_Testing_candidates, mixture_PI_candidates, mixture_sigma1_candidates, mixture_sigma2_candidates)
    
    for LearningRate, N_Samples_Testing, pi, sigma1, sigma2 in hyper_list:
      
      print("*"*50)
      
      print("Learning rate: {}".format(LearningRate))
      print("N_Samples_Testing: {}".format(N_Samples_Testing))
      
      # Initialize network
      net = BayesianNN(
        nn_input_size=784, 
        layer_config=[400, 400, 10], 
        activation_config=[ActivationType.RELU, ActivationType.RELU, ActivationType.SOFTMAX], 
        prior_type=PriorType.MIXTURE,
        prior_params={'pi' : pi, 'sigma1' : sigma1, 'sigma2' : sigma2},
        task_type=TaskType.CLASSIFICATION
      )

      if use_cuda:
        net = net.cuda()

      optim = torch.optim.SGD(net.parameters(), lr=LearningRate)

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
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()

              y_pred = net(batch_X)

              # Loss and backprop
              loss, kl, _ = net.cost_function(batch_X, batch_Y, num_samples=2, num_batches = N_Train_Batch)
              optim.zero_grad()
              loss.backward()
              optim.step()

          # Evaluate on training set
          # net.eval()
          train_X = Variable(raw_train_set.train_data[train_indices].view(train_size, -1).type(torch.FloatTensor))
          train_Y = Variable(raw_train_set.train_labels[train_indices].view(train_size, -1))

          if use_cuda:
            train_X, train_Y = train_X.cuda(), train_Y.cuda()

          pred_class = net.predict_by_sampling(train_X, num_samples=N_Samples_Testing).data.cpu().numpy().argmax(axis=1)
          true_class = train_Y.data.cpu().numpy().ravel()

          train_accu = compute_accu(pred_class, true_class, 2)
          print('Epoch', i_ep, '|  Training Accuracy:', train_accu, '%')

          train_accu_lst.append(train_accu)

          # Evaluate on test set
          test_X = Variable(raw_train_set.train_data[valid_indices].view(valid_size, -1).type(torch.FloatTensor))
          test_Y = Variable(raw_train_set.train_labels[valid_indices].view(valid_size, -1))

          if use_cuda:
            test_X, test_Y = test_X.cuda(), test_Y.cuda()

          pred_class = net.predict_by_sampling(test_X, num_samples=N_Samples_Testing).data.cpu().numpy().argmax(axis=1)
          true_class = test_Y.data.cpu().numpy().ravel()

          test_accu = compute_accu(pred_class, true_class, 2)
          print('Epoch', i_ep, '|  Valid Accuracy:', test_accu, '%')

          test_accu_lst.append(test_accu)

      # to report the final test error, I will use the average of test errors of the last 10 epochs
      report_test_error_mean = np.average(test_accu_lst[-10:])
      report_test_error_std = np.std(test_accu_lst[-10:])

      # Plot
#       import matplotlib.pyplot as plt
#       plt.style.use('seaborn-paper')

#       plt.title('Classification Accuracy on MNIST')
#       plt.plot(train_accu_lst, label='Train')
#       plt.plot(test_accu_lst, label='Test')
#       plt.ylabel('Accuracy (%)')
#       plt.xlabel('Epochs')
#       plt.legend(loc='best')
#       plt.tight_layout()
#       plt.show()
      print("Final report test error: {} +- {}".format(report_test_error_mean, report_test_error_std))
      
      hyper_val_error_dict[(LearningRate, N_Samples_Testing, pi, sigma1, sigma2)] = (report_test_error_mean, report_test_error_std)
      for key in hyper_val_error_dict.keys():
        print("{}: {}".format(key, hyper_val_error_dict[key]))
