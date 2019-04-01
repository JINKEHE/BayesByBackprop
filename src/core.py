import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch import optim
from torch import nn
from torch import distributions as dist
from enum import Enum

# boolean variable that indicates whether or not we have gpu...
use_cuda = torch.cuda.is_available()
print("Use GPU: {}".format(use_cuda))

# Default gaussian mixture parameters
PI = 0.5

SIGMA_1 = torch.tensor([math.exp(-0)])
SIGMA_2 = torch.tensor([math.exp(-6)])

# place tensor in GPU if use_cuda
if use_cuda:
  SIGMA_1 = SIGMA_1.cuda()
  SIGMA_2 = SIGMA_2.cuda()

# Default gaussian parameters
MU_PRIOR = 0
SIGMA_PRIOR = torch.tensor([math.exp(-0)])

# place tensor in GPU if use_cuda
if use_cuda:
    SIGMA_PRIOR = SIGMA_PRIOR.cuda()

# Initial weight hyperparameters
MU_WEIGHTS = (-0.03, 0.03)
RHO_WEIGHTS = (-2.31, -2.30)
MU_BIAS = (-0.03, 0.03)
RHO_BIAS = (-2.31, -2.30)

# Loss variance
SIGMA = torch.tensor([math.exp(-2)])

# place tensor in GPU if use_cuda
if use_cuda:
    SIGMA = SIGMA.cuda()

class PriorType(Enum):
  MIXTURE = 1
  GAUSSIAN = 2

class ActivationType(Enum):
  NONE = 0
  RELU = 1
  SOFTMAX = 2
  TANH = 3
  SIGMOID = 4

class TaskType(Enum):
  REGRESSION = 1
  CLASSIFICATION = 2

class GaussianMixture(object):

  def __init__(self, pi, sigma1, sigma2):
    self.pi = pi
    self.sigma1 = sigma1
    self.sigma2 = sigma2

  # arguments of dist.Normal() should be tensors rather than scalars
  def log_prob(self, weights):
    new_weights = weights.view(-1)
    normal_density1 = dist.Normal(0,self.sigma1).log_prob(new_weights)
    exp_normal_density1 = torch.exp(normal_density1)
    exp_normal_density2 = torch.exp(
        dist.Normal(0.0, self.sigma2).log_prob(new_weights))
    nonzero = exp_normal_density2.nonzero()
    zero = (exp_normal_density2==0).nonzero()
    sum_log_prob = torch.sum(torch.log(self.pi * torch.take(exp_normal_density1,nonzero) \
                  + (1-self.pi)*torch.take(exp_normal_density2,nonzero))) \
                  + torch.sum(torch.take(normal_density1, zero)+np.log(self.pi))
    return sum_log_prob

class BayesianLayer(nn.Module):

  def __init__(self,
               input_size,
               output_size,
               prior_type=PriorType.MIXTURE,
               prior_params={'pi' : PI, 'sigma1' : SIGMA_1, 'sigma2' : SIGMA_2},
               activation_type=ActivationType.NONE,
              ):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.activation_type = activation_type

    # create torch variables
    if not use_cuda:
        self.mu_weights = nn.Parameter(torch.Tensor(output_size, input_size))
        self.rho_weights = nn.Parameter(torch.Tensor(output_size, input_size))
        self.mu_bias = nn.Parameter(torch.Tensor(output_size))
        self.rho_bias = nn.Parameter(torch.Tensor(output_size))
        self.normal_dist = dist.Normal(torch.Tensor([0]), torch.Tensor([1]))
    else:
        self.mu_weights = nn.Parameter(torch.Tensor(output_size, input_size).cuda())
        self.rho_weights = nn.Parameter(torch.Tensor(output_size, input_size).cuda())
        self.mu_bias = nn.Parameter(torch.Tensor(output_size).cuda())
        self.rho_bias = nn.Parameter(torch.Tensor(output_size).cuda())
        self.normal_dist = dist.Normal(torch.Tensor([0]).cuda(), torch.Tensor([1]).cuda())

    # initialize variables
    self.mu_weights.data.uniform_(*MU_WEIGHTS)
    self.rho_weights.data.uniform_(*RHO_WEIGHTS)
    self.mu_bias.data.uniform_(*MU_BIAS)
    self.rho_bias.data.uniform_(*RHO_BIAS)

    if prior_type == PriorType.MIXTURE:
      self.prior_weights = GaussianMixture(
          prior_params['pi'], prior_params['sigma1'], prior_params['sigma2'])
      self.prior_bias = GaussianMixture(
          prior_params['pi'], prior_params['sigma1'], prior_params['sigma2'])
    else:
      self.prior_weights = dist.Normal(prior_params['mean'],
                                       prior_params['sigma'])
      self.prior_bias = dist.Normal(prior_params['mean'],
                                    prior_params['sigma'])
    self.log_prior = 0
    self.log_posterior = 0

  def _compute_gaussian_sample(self, mu, rho):
    epsilon = self.normal_dist.sample(rho.size()).squeeze(-1)
    return mu + torch.log2(1 + torch.exp(rho)) * epsilon

  def forward(self, input_data, sample=False, debug=False):
    if self.training or sample:
      weights = self._compute_gaussian_sample(self.mu_weights, self.rho_weights)
      bias = self._compute_gaussian_sample(self.mu_bias, self.rho_bias)
      if debug is True:
        print("sampled weights:")
        print(weights)
      self.log_prior = (self.prior_weights.log_prob(weights).sum() +
                        self.prior_bias.log_prob(bias).sum() )
      sigma_weights = torch.log(1 + torch.exp(self.rho_weights))
      sigma_bias = torch.log(1 + torch.exp(self.rho_bias))
      self.log_posterior = (
          dist.Normal(
              self.mu_weights, sigma_weights).log_prob(weights).sum() +
          dist.Normal(self.mu_bias, sigma_bias).log_prob(bias).sum()
      )

      if torch.isnan(self.log_posterior):
        print('Weights log prob: ')
        print( dist.Normal(
              self.mu_weights, sigma_weights).log_prob(weights).sum())
        print('Bias log prob: ' )
        print(dist.Normal(self.mu_bias, sigma_bias).log_prob(bias).sum())
    else:
      weights = self.mu_weights
      bias = self.mu_bias

    linear_output = nn.functional.linear(input_data, weights, bias)
    output = linear_output
    if self.activation_type == ActivationType.RELU:
      output = torch.relu(linear_output)
    elif self.activation_type == ActivationType.SOFTMAX:
      output = torch.log_softmax(linear_output, dim=1)
    elif self.activation_type == ActivationType.SIGMOID:
      output = torch.sigmoid(linear_output)
    elif self.activation_type == ActivationType.TANH:
      output = torch.tanh(linear_output)
    elif self.activation_type == ActivationType.NONE:
      output = linear_output
    else:
      raise ValueError('activation_type {} not support'.format(self.activation_type))
    return output

  def extra_repr(self):
    return 'Bayesian Layer, in_size:{}, out_size:{}, activation_type:{}'.format(
      self.input_size, self.output_size, self.activation_type.name
    )

class BayesianNN(nn.Module):

  def __init__(
      self,
      nn_input_size,
      layer_config=[100, 100, 10],           # list of layer output sizes
      activation_config=[ActivationType.RELU, ActivationType.RELU, ActivationType.NONE],
      prior_type=PriorType.MIXTURE,
      prior_params={'pi' : PI, 'sigma1' : SIGMA_1, 'sigma2' : SIGMA_2},
      task_type=TaskType.REGRESSION,         # determines the likelihood form
  ):
    super().__init__()

    self.layers = nn.ModuleList([]) # ensures that all params are registered
    self.input_size = nn_input_size
    for i, output_size in enumerate(layer_config):
      if i == 0:
        input_size = self.input_size
      else:
        input_size = layer_config[i-1]

      bayesian_layer = BayesianLayer(input_size, output_size,
                                     activation_type = activation_config[i],
                                     prior_type=prior_type,
                                     prior_params=prior_params)
      self.layers.append(bayesian_layer)
    self.output_size = self.layers[-1].output_size
    self.task_type = task_type

  def forward(self, input_data, sample=True, debug=False):
    current_data = input_data
    for layer in self.layers:
      current_data = layer.forward(current_data, sample, debug=debug)
    if sample is False:
        print("not sampling.")
    return current_data

  # sample a bunch of weights for the network
  # make predictions using sampled weights
  # output averaged predictions from different sampled weights
  def predict_by_sampling(self, input_data, num_samples=1):
    # reduce the use of memory
    with torch.no_grad():
        outputs = torch.empty(num_samples, input_data.size()[0], self.output_size)
        for i in range(num_samples):
            print("*"*20)
            outputs[i] = self.forward(input_data, sample=True, debug=True)
            print(outputs[i][0])
            print("*"*20)
        stds = outputs.std(0)
        print("std in probability distributions:", stds.mean(0))
        outputs = outputs.mean(0)
    return outputs

  def log_prior(self):
    log_prior = 0
    for layer in self.layers:
      log_prior += layer.log_prior
    return log_prior

  def log_posterior(self):
    log_posterior = 0
    for layer in self.layers:
      log_posterior += layer.log_posterior
    return log_posterior

  def cost_function(self, inputs, targets, num_samples, ratio):
    sum_log_posterior = 0
    sum_log_prior = 0
    sum_negative_log_likelihood = 0
    for n in range(num_samples):
      outputs = self(inputs, sample=True)
      sum_log_posterior += self.log_posterior()
      sum_log_prior += self.log_prior()
      if self.task_type == TaskType.CLASSIFICATION:
         # the outputs are from log_softmax activation function
         log_probs = outputs[range(targets.size()[0]), targets]
         # the negative sum of log probs is the negative log likelihood
         # for this sampled neural network on this minibatch
         negative_log_likelihood = -log_probs.sum()
         # negative_log_likelihood = nn.functional.nll_loss(outputs, targets)
      elif self.task_type == TaskType.REGRESSION:
         negative_log_likelihood = - dist.Normal(
             targets, SIGMA).log_prob(outputs).sum()
      sum_negative_log_likelihood += negative_log_likelihood
    kl_divergence = (sum_log_posterior / num_samples - sum_log_prior / num_samples) * ratio
    negative_log_likelihood = sum_negative_log_likelihood / num_samples
    loss =  kl_divergence + negative_log_likelihood
    return loss, kl_divergence, negative_log_likelihood

  def extra_repr(self):
    repr = ''
    for layer in self.layers:
      repr += layer.extra_repr()
      repr += '\n'
    return repr
