import queue
import random
import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data
from torch import optim
from torch import nn
from torch import distributions as dist

from core import *


use_cuda = torch.cuda.is_available()

CONTEXT_SIZE = 117 + 1
SAMPLE_COUNT = 2
NUM_ACTIONS = 2
AGENT_MEMORY_LEN = 4096

EDIBLE_REWARD = 5.0
POISONOUS_REWARD = -35.0

EDIBLE_CONSTANT = 1.0
POISONOUS_CONSTANT = -1.0

'''The environment sends the agent the features of a mushroom.

The agent can pick one of two actions:
1. eat
2. pass

If the agent picks pass, the regret is always 0.

If the agent picks eat:
1. if the mushroom is edible, the reward is always +5.0.
2. if the mushroom is poisonous, with prob 1/2 the reward is -35.0, with prob 1/2 the reward is +5.0'''

'''The system is represented by two entities: an **agent** and an **environment**.  They interact in rounds, in the following manner:

1. The environment randomly selects a mushroom from the dataset and presents its features (the current context) to the agent.

2. The agent selects an action that it deems optimal given the context.

3. The environment computes the reward for the selected action and sends it back to the agent, which updates its predictions.'''

class Agent(object):  
  def __init__(self):
    # Previous 4096 interactions with the environment
    self.past_plays_context = []
    self.past_plays_action = []
    self.past_plays_reward = []
    self.value_estimates = None
    
    
  def collected_data_count(self):
    return len(self.past_plays_context)

  def select_action(self, context, logs):
    pass
  def update_variational_posterior(self, logs):
    pass
  def update_memory(self, context, action, reward):
    self.past_plays_context.append(context)
    self.past_plays_action.append(action)
    self.past_plays_reward.append(reward)
    if len(self.past_plays_context) == AGENT_MEMORY_LEN:
      self.past_plays_context = self.past_plays_context[1:]
      self.past_plays_action = self.past_plays_action[1:]
      self.past_plays_reward = self.past_plays_reward[1:]
    

class EGreedyNNAgent(Agent):
  def __init__(self, epsilon, optimizer_constructor=torch.optim.Adam, optim_params={'lr':1e-3, 'eps':0.01}):
    super().__init__()
    self.epsilon = epsilon
    self.value_estimates =  nn.Sequential(
        torch.nn.Linear(CONTEXT_SIZE, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
    )
    
    self.loss_fn = nn.MSELoss()
    
    if use_cuda:
      self.value_estimates = self.value_estimates.cuda()

    self.optimizer = optimizer_constructor(self.value_estimates.parameters(), **optim_params)
    
    
  def select_action(self, context, logs=False):
    if random.random() < self.epsilon:
      return random.randint(0,NUM_ACTIONS-1)
    
    max_reward = POISONOUS_REWARD - 1
    argmax_action = -1
    for action in range(NUM_ACTIONS):
      estimated_reward = 0
      
        
      action_tensor = torch.tensor([[action]], dtype=torch.float)
        
      context_and_action = torch.cat(
          [context, action_tensor], dim=1)
        
      if use_cuda:
        context_and_action = context_and_action.cuda()
        
      estimated_reward = self.value_estimates(context_and_action)
      if logs:
        print('Action {} - predicted reward: {}'.format(
            action, estimated_reward))
      if estimated_reward > max_reward:
        max_reward = estimated_reward
        argmax_action = action
    return argmax_action

  
  def update_variational_posterior(self, logs=False):
    features = []
    for context, action in zip(iter(self.past_plays_context),
                               iter(self.past_plays_action)):
      
      action_tensor = torch.tensor([[action]], dtype=torch.float)
      
      features.append(torch.cat(
          [context, action_tensor], dim=1))
    features = torch.cat(features)
    
    rewards = torch.tensor(self.past_plays_reward, dtype=torch.float)

    past_plays_set = torch.utils.data.TensorDataset(features, rewards)
    past_plays_loader = torch.utils.data.DataLoader(
        past_plays_set, batch_size=64, shuffle=True, num_workers=1)
    
    avg_loss = 0
    
    for i, data in enumerate(past_plays_loader):
      inputs, labels = data
      if use_cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
      # zero the parameter gradients
      self.optimizer.zero_grad()

      # forward + backward + optimize
      loss = self.loss_fn(self.value_estimates(inputs).squeeze(), labels)

      loss.backward()
      self.optimizer.step()
     
      avg_loss += loss
      
    avg_loss /= len(past_plays_loader.dataset)
    
    if logs:
      print('{}. Loss: {}'.format(i, avg_loss))
    return avg_loss    
  

class BNNAgent(Agent):
  
  def __init__(self, optimizer_constructor=torch.optim.Adam, 
               optim_params={'lr':1e-3, 'eps':0.01},
               prior_params=None,
               lr_scheduler_step_size=32):
        
    super().__init__()
    
    # Neural net for estimating E[reward | context, action]
    
    bnn_params = {'nn_input_size': CONTEXT_SIZE,
                  'layer_config': [100, 100, 1],
                  'activation_config': [ActivationType.RELU, 
                                        ActivationType.RELU, ActivationType.NONE],
                 }
    
    if prior_params is not None:
        bnn_params['prior_params'] = prior_params
    
    self.value_estimates = BayesianNN(**bnn_params)
        
    if use_cuda:
      self.value_estimates = self.value_estimates.cuda()
    
    self.optimizer = optimizer_constructor(self.value_estimates.parameters(), **optim_params)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_scheduler_step_size)
  

  def select_action(self, context, logs=False):
    self.value_estimates.train()
    max_reward = POISONOUS_REWARD - 1
    argmax_action = -1
    for action in range(NUM_ACTIONS):
      expected_reward = 0
      for i in range(SAMPLE_COUNT):
        
        action_tensor = torch.tensor([[action]], dtype=torch.float)
        
        context_and_action = torch.cat(
            [context, action_tensor], dim=1)
        
        if use_cuda:
          context_and_action = context_and_action.cuda()
        
        expected_reward += self.value_estimates(context_and_action)
      expected_reward /= SAMPLE_COUNT
      if logs:
        print('Action {} - predicted reward: {}'.format(
            action, expected_reward))
      if expected_reward > max_reward:
        max_reward = expected_reward
        argmax_action = action
    return argmax_action
  

  def update_variational_posterior(self, logs=False):
    features = []
    for context, action in zip(iter(self.past_plays_context),
                               iter(self.past_plays_action)):
      
      action_tensor = torch.tensor([[action]], dtype=torch.float)
      
      features.append(torch.cat(
          [context, action_tensor], dim=1))
    features = torch.cat(features)
    
    rewards = torch.tensor(self.past_plays_reward, dtype=torch.float)

    past_plays_set = torch.utils.data.TensorDataset(features, rewards)
    past_plays_loader = torch.utils.data.DataLoader(
        past_plays_set, batch_size=64, shuffle=True, num_workers=1)
    
    avg_loss = 0
    
    for i, data in enumerate(past_plays_loader):
      inputs, labels = data
      if use_cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
      # zero the parameter gradients
      self.optimizer.zero_grad()

      # forward + backward + optimize
      loss, _, _ = self.value_estimates.cost_function(
          inputs, labels, num_samples=2, num_batches=len(past_plays_loader))
      loss.backward()
      self.optimizer.step()
     
      avg_loss += loss
      
    avg_loss /= len(past_plays_loader.dataset)
#     self.scheduler.step()
    
    if logs:
      print('{}. Loss: {}'.format(i, avg_loss))
    return avg_loss    
      

class Environment(object):
  
  def __init__(self, agent, dataloader):
    self.agent = agent
    self.dataloader = dataloader
    self.cumulative_regret = 0
  
  def play_round(self, logs=False):
    
    # Get the features and label of a random mushroom
    context, mushroom_type = next(iter(self.dataloader))
    
    # Set GPU tensors; context doesn't get updated because it is fed into
    # the dataloader of the agent
    if use_cuda:
      mushroom_type = mushroom_type.cuda()
    
    # Determine the 'eat' reward and the optimal reward for the current mushroom
    if mushroom_type == EDIBLE_CONSTANT:
      eat_reward = EDIBLE_REWARD
      optimal_reward = EDIBLE_REWARD
      mushroom_string = 'edible'
    else: #poisonous
      optimal_reward = 0.0
      mushroom_string = 'poisonous'
      random_draw = random.random()
      if random_draw > 0.5:
        eat_reward = EDIBLE_REWARD
      else:
        eat_reward = POISONOUS_REWARD
        
    # Present the context to the agent and get the the agent's action of choice
    selected_action = self.agent.select_action(context, logs)
    
    # Determine the reward for the agent's action and the regret
    if selected_action == 0: #not eat
      action_string = 'pass'
      reward = 0
    else: #eat
      action_string = 'eat'
      reward = eat_reward
    self.cumulative_regret += max(optimal_reward - reward, 0) 
    
    if logs:
      print('The mushroom was {}. The agent chose {} and got a reward of {}.'.format(mushroom_string, action_string, reward))
      print('Cumulative regret is {}'.format(self.cumulative_regret))
      
    # Send the reward information back to the agent for logging and variational
    # posterior update
    self.agent.update_memory(context, selected_action, reward)
    loss = self.agent.update_variational_posterior(logs)
    
    return loss.item()
