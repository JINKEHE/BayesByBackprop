# -*- coding: utf-8 -*-
"""Bandits.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ryV2BfuI6A-J3g7UzzGreDlS6onh3HPw
"""

import queue
import random
import torch
import copy
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data
from torch import optim
from torch import nn
from torch import distributions as dist

sys.path.append('..')
from core import *
from bandits import *

print(torch.__version__)

use_cuda = torch.cuda.is_available()

CONTEXT_SIZE = 117 + 1
SAMPLE_COUNT = 2
NUM_ACTIONS = 2
AGENT_MEMORY_LEN = 4096

EDIBLE_REWARD = 5.0
POISONOUS_REWARD = -35.0

EDIBLE_CONSTANT = 1.0
POISONOUS_CONSTANT = -1.0

"""The environment sends the agent the features of a mushroom.

The agent can pick one of two actions:
1. eat
2. pass

If the agent picks pass, the regret is always 0.

If the agent picks eat:
1. if the mushroom is edible, the reward is always +5.0.
2. if the mushroom is poisonous, with prob 1/2 the reward is -35.0, with prob 1/2 the reward is +5.0

The system is represented by two entities: an **agent** and an **environment**.  They interact in rounds, in the following manner:

1. The environment randomly selects a mushroom from the dataset and presents its features (the current context) to the agent.

2. The agent selects an action that it deems optimal given the context.

3. The environment computes the reward for the selected action and sends it back to the agent, which updates its predictions.
"""

def read_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str, default='Experiment')
  parser.add_argument('--optimizer_type', type=str, default='Adam')
  parser.add_argument('--eg_learning_rate', type=float, default=1e-3)
  parser.add_argument('--eg_epsilon', type=float, default=1e-3)
  parser.add_argument('--bnn_learning_rate', type=float, default=1e-3)
  parser.add_argument('--bnn_epsilon', type=float, default=1e-3)
  parser.add_argument('--bnn_lr_scheduler_step_size', type=int, default=32)
  parser.add_argument('--bnn_pi', type=float, default=0.75)
  parser.add_argument('--bnn_log_sigma1', type=float, default=math.exp(-2))
  parser.add_argument('--bnn_log_sigma2', type=float, default=math.exp(-7))
  parser.add_argument('--averaged_weights', dest='averaged_weights', action='store_true')
  parser.add_argument('--avg_weights_count', type=int, default=2)
  args = parser.parse_args()
  return args
  

if __name__ == '__main__':

	# Load the UCI Mushroom Dataset: 8124 datapoints, each with 22 categorical
	# features and one label - edible/poisonous. The features are transformed to a
	# one-hot encoding. 
	# The missing values (marked with ?) are treated as a different class for now.

  mushroom_dataset = pd.read_csv('mushrooms.csv')
  train_labels = mushroom_dataset['class']
  train_labels = train_labels.replace(['p', 'e'],
                                [POISONOUS_CONSTANT, EDIBLE_CONSTANT])
  train_features = pd.get_dummies(mushroom_dataset.drop(['class'], axis=1))

  train_features = torch.tensor(train_features.values, dtype=torch.float)
  train_labels = torch.tensor(train_labels.values)

  trainset = torch.utils.data.TensorDataset(train_features, train_labels)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=True, num_workers=0)
    
	# Parameters for bnn and eg agents
  args = read_args(sys.argv)
  
  if args.optimizer_type == 'Adam':
    optimizer_constructor = torch.optim.Adam
    eg_optimizer_params = {'lr': args.eg_learning_rate,
                           'eps': args.eg_epsilon}
    bnn_optimizer_params = {'lr': args.bnn_learning_rate,
                            'eps': args.bnn_epsilon}
  
  sigma1 = math.exp(args.bnn_log_sigma1)
  sigma2 = math.exp(args.bnn_log_sigma2)
  prior_params = {'pi': args.bnn_pi, 
                  'sigma1': sigma1,
                  'sigma2': sigma2}
	      
  bnn_agent = BNNAgent(optimizer_constructor=optimizer_constructor,
	               optim_params=bnn_optimizer_params,
	               prior_params=prior_params,
	               lr_scheduler_step_size=args.bnn_lr_scheduler_step_size,
                 averaged_weights=args.averaged_weights,
                 avg_weights_count=args.avg_weights_count)
  bnn_env = Environment(bnn_agent, trainloader)

  eg5_agent = EGreedyNNAgent(epsilon=.05, 
	                    optimizer_constructor=optimizer_constructor,
	                    optim_params=eg_optimizer_params)
  eg5_env = Environment(eg5_agent, copy.deepcopy(trainloader))

  eg1_agent = EGreedyNNAgent(epsilon=.01, 
	                    optimizer_constructor=optimizer_constructor,
	                    optim_params=eg_optimizer_params)
  eg1_env = Environment(eg1_agent, copy.deepcopy(trainloader))

  eg0_agent = EGreedyNNAgent(epsilon=.00, 
	                    optimizer_constructor=optimizer_constructor,
	                    optim_params=eg_optimizer_params)
  eg0_env = Environment(eg0_agent, copy.deepcopy(trainloader))

  eg5_regret = []
  eg1_regret = []
  eg0_regret = []
  bnn_loss = []
  bnn_regret = []

  # If necessary, create directory for graph outputs
  if not os.path.isdir('results'):
    os.makedirs('results')  

  if not os.path.isdir('results/{}'.format(args.experiment_name)):
    os.makedirs('results/{}'.format(args.experiment_name))

  if not os.path.isdir('results/{}/graphs'.format(args.experiment_name)):
    os.makedirs('results/{}/graphs'.format(args.experiment_name))

  for i in range(4000):

    logs = False
    if (i+1) % 100 == 0:
      logs = True
      print('{}.'.format(i))

    eg5_env.play_round(logs=logs)
    eg1_env.play_round(logs=logs)
    eg0_env.play_round(logs=logs)
    current_bnn_loss = bnn_env.play_round(logs=logs)

    if (i+1) % 50 == 0:
      bnn_loss.append(current_bnn_loss)
    eg5_regret.append(eg5_env.cumulative_regret)
    eg1_regret.append(eg1_env.cumulative_regret)
    eg0_regret.append(eg0_env.cumulative_regret)
    bnn_regret.append(bnn_env.cumulative_regret)

    if (i+1) % 500 == 0:
      plt.plot(np.array(bnn_loss), label='BNN loss')
      plt.legend()
      plt.ylabel('Loss')
      plt.savefig('results/{}/graphs/bnn_loss_{}'.format(args.experiment_name, i+1))
      plt.clf()
      
      plt.plot(np.array(eg5_regret), label='Epsilon-Greedy 5% Regret')
      plt.plot(np.array(eg1_regret), label='Epsilon-Greedy 1% Regret')
      plt.plot(np.array(eg0_regret), label='Greedy Regret')
      plt.plot(np.array(bnn_regret), label='BNN Regret')
      plt.legend()
      plt.ylabel('Cumulative Regret')
      plt.savefig('results/{}/graphs/regret_{}'.format(args.experiment_name, i+1))
      plt.clf()
      bnn_loss = []
      eg_loss = []
