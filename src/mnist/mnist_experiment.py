import sys
sys.path.append("..")
from core import *
import torch
from torchvision import transforms
import torchvision
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
import random
import math
import itertools
import argparse
import pickle
import torch.nn.functional as f

# check whether we are using GPU or not
use_cuda = torch.cuda.is_available()

def get_args(args=None):
    # the basic arguments for MNIST experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_weights', dest="save_weights", action="store_true")
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--num_units', type=int, help='the number of units in the hidden layer', default=1200)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--network_type', type=str, default='standard')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    # for standard nerual networks
    parser.add_argument('--use_dropout', dest='use_dropout', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    # for bayesian neural networks
    parser.add_argument('--num_samples_training', type=int, default=2)
    parser.add_argument('--num_samples_testing', type=int, default=10)
    # for gaussian prior
    parser.add_argument('--gaussian_mean', type=float, default=0.0)
    parser.add_argument('--gaussian_log_sigma', type=float, default=-0.0)
    # for scale mixture gaussian prior
    parser.add_argument('--prior_type', type=str, default='scale_mixture')
    parser.add_argument('--scale_mixture_pi', type=float, default=0.5)
    parser.add_argument('--scale_mixture_log_sigma1', type=float, default=-0.0)
    parser.add_argument('--scale_mixture_log_sigma2', type=float, default=-6.0)
    parser.add_argument('--lr_scheduler_step_size', type=int, default=1000)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--initial_mu_weights', type=float, nargs=2, default=[-0.03,0.03])
    parser.add_argument('--initial_rho_weights', type=float, nargs=2, default=[-4,-2])
    parser.add_argument('--initial_mu_bias', type=float, nargs=2, default=[-0.03,0.03])
    parser.add_argument('--initial_rho_bias', type=float, nargs=2, default=[-4,-2])
    # for minibatch training
    parser.add_argument('--use_normalized', dest='use_normalized', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # read experiment arguments
    args = get_args(sys.argv)
    LearningRate = args.lr
    save_weights = args.save_weights
    experiment_name = args.experiment_name
    N_Epochs = args.num_epochs
    BatchSize = args.batch_size
    N_units = args.num_units
    network_type = args.network_type
    lr_scheduler_step_size = args.lr_scheduler_step_size
    lr_scheduler_gamma = args.lr_scheduler_gamma
    assert network_type == 'standard' or network_type == 'bayesian'
    optimizer_type = args.optimizer
    preprocess = args.preprocess
    use_normalized = args.use_normalized
    print("Experiment Settings")
    print("num epochs: {}".format(N_Epochs))
    print("batch size: {}".format(BatchSize))
    print("learning rate: {}".format(LearningRate))
    print("num units: {}".format(N_units))
    print("network type: {}".format(network_type))
    print("optimizer type: {}".format(optimizer_type))
    print("divide pixels by 126: {}".format(preprocess))

    # build the neural network
    if network_type == "standard":
        use_dropout = args.use_dropout
        dropout_rate = args.dropout_rate
        if use_dropout:
            print("dropout rate: {}".format(dropout_rate))
        else:
            print("use dropout: False")
    else:
        N_Samples_Training = args.num_samples_training
        N_Samples_Testing = args.num_samples_testing
        use_normalized = args.use_normalized
        print("training sample size: {}".format(N_Samples_Training))
        print("testing sample size: {}".format(N_Samples_Testing))
        print("use normalized:", use_normalized)
        prior_type = args.prior_type
        if prior_type == "scale_mixture":
            mixture_sigma1 = torch.tensor([math.exp(args.scale_mixture_log_sigma1)])
            mixture_sigma2 = torch.tensor([math.exp(args.scale_mixture_log_sigma2)])
            if use_cuda:
                mixture_sigma1 = mixture_sigma1.cuda()
                mixture_sigma2 = mixture_sigma2.cuda()
            mixture_pi = args.scale_mixture_pi
            print("prior distribution: scale mixture ({}, {}, {})".format(mixture_pi, mixture_sigma1, mixture_sigma2))
        elif prior_type == "gaussian":
            gaussian_mean = torch.tensor([args.gaussian_mean])
            gaussian_sigma = torch([math.exp(args.gaussian_log_sigma)])
            if use_cuda:
                gaussian_mean = gaussian_mean.cuda()
                gaussian_sigma = gaussian_sigma.cuda()
            print("prior distribution: gaussian (mean: {}, sigma: {})".format(gaussian_mean, gaussian_sigma))

    if experiment_name is None:
        experiment_name = "mnist_num_units_{}_optim_{}".format(N_units, optimizer_type)
        if network_type == 'standard':
            if use_dropout:
                experiment_name += "_standard_dropout_{}".format(dropout_rate)
            else:
                experiment_name += "_standard_SGD"
        elif network_type == 'bayesian':
            # maybe also add prior distribution parameters to the file name?
            experiment_name += "_bayesian_{}_num_samples_testing_{}".format(prior_type, N_Samples_Testing)
    print(experiment_name)

    # load the dataset
    import os.path
    dataset_path = os.path.join(os.path.dirname(""), 'mnist_dataset')
    if not os.path.exists(dataset_path):
        Download_MNIST = True
    else:
        Download_MNIST = False

    if preprocess is True:
        # transform = transforms.Compose([np.array, torch.FloatTensor, lambda x: torch.div(x,126.0)])
        transform = transforms.Compose([torchvision.transforms.ToTensor(), lambda x: x*255/126])
    else:
        transform = transforms.Compose([np.array, torch.FloatTensor])

    train_set = torchvision.datasets.MNIST(
        root=dataset_path,
        train=True,
        transform=transform,
        download=Download_MNIST
    )

    train_loader = Data.DataLoader(dataset=train_set, batch_size=BatchSize, shuffle=True)

    test_set = torchvision.datasets.MNIST(
        root=dataset_path,
        train=False,
        transform=transform,
        download=Download_MNIST
    )

    train_size = train_set.train_data.size()[0]
    N_Train_Batch = train_size / BatchSize
    test_size = test_set.test_data.size()[0]
    compute_accu = lambda pred, true, digits: round((pred == true).mean() * 100, digits)

    # build the network
    if network_type == "standard":
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        if use_dropout:
            dropout_rate = args.dropout_rate
            net =  torch.nn.Sequential(
                torch.nn.Linear(784, N_units),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(N_units, N_units),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(N_units, 10),
                torch.nn.LogSoftmax(dim=1)
            )
        else:
            net = torch.nn.Sequential(
                torch.nn.Linear(784, N_units),
                torch.nn.ReLU(),
                torch.nn.Linear(N_units, N_units),
                torch.nn.ReLU(),
                torch.nn.Linear(N_units, 10),
                torch.nn.LogSoftmax(dim=1)
            )
    elif network_type == "bayesian":
        if prior_type == 'scale_mixture':
            prior_type = PriorType.MIXTURE
            prior_params={'pi' : mixture_pi, 'sigma1' : mixture_sigma1, 'sigma2' : mixture_sigma2}
        else:
            prior_type = PriorType.GAUSSIAN
            prior_params={'mean': gaussian_mean, 'sigma': gaussian_sigma}
        net = BayesianNN(
            nn_input_size=784,
            layer_config=[N_units, N_units, 10],
            activation_config=[ActivationType.RELU, ActivationType.RELU, ActivationType.SOFTMAX],
            prior_type=prior_type,
            prior_params=prior_params,
            task_type=TaskType.CLASSIFICATION,
            initial_mu_weights=args.initial_mu_weights,
            initial_rho_weights=args.initial_rho_weights,
            initial_mu_bias=args.initial_mu_bias,
            initial_rho_bias=args.initial_rho_bias
        )
        print("initial mu weights:", args.initial_mu_weights)
        print("initial rho weights:", args.initial_rho_weights)
        print("initial mu bias:", args.initial_mu_bias)
        print("initial rho bias", args.initial_rho_bias)
    else:
        raise ValueError

    if use_cuda:
        net = net.cuda()

    print(net)

    # build the optimizer
    if optimizer_type == "SGD":
        optim = torch.optim.SGD(net.parameters(), lr=LearningRate)
    elif optimizer_type == "Adam":
        optim = torch.optim.Adam(net.parameters(), lr=LearningRate)
    else:
        raise ValueError

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
    # the main training loop
    train_accu_lst = []
    test_accu_lst = []

    test_X = Variable(test_set.test_data.view(test_size, -1).type(torch.FloatTensor))
    test_Y = Variable(test_set.test_labels.view(test_size, -1))

    if use_cuda:
        test_X, test_Y = test_X.cuda(), test_Y.cuda()

    train_X = Variable(train_set.train_data.view(train_size, -1).type(torch.FloatTensor))
    train_Y = Variable(train_set.train_labels.view(train_size, -1))

    if use_cuda:
        train_X, train_Y = train_X.cuda(), train_Y.cuda()

    normalized_factor = 1/N_Train_Batch
    for i_ep in range(N_Epochs):
        scheduler.step()
        # Training
        net.train()
        if use_normalized:
            normalized_factor = 1
        total_loss = 0
        total_kl = 0
        for X, Y in train_loader:
            batch_X = Variable(X.view(X.size()[0], -1))
            batch_Y = Variable(Y.view(X.size()[0]))

            if use_normalized:
                normalized_factor /= 2
            if use_cuda:
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()

            # compute loss
            if network_type == 'standard':
                y_pred = net(batch_X)
                loss = loss_fn(y_pred, batch_Y)
            elif network_type == 'bayesian':
                loss, kl , _ = net.cost_function(batch_X, batch_Y, num_samples=N_Samples_Training, ratio = normalized_factor)
                total_loss += loss.item()
                total_kl += kl.item()
                # detect nan
                if torch.isnan(loss):
                    print("Loss NAN.")
                    for p in net.parameters():
                        print(p)
                    raise ValueError
            else:
                raise ValueError

            # do backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Evaluate on training set
        if network_type == 'standard':
            net.eval()
        # do not use evaluation mode for bayesian networks because we do sampling during testing
        if network_type == 'bayesian':
            print(total_loss, total_kl)

        # test on training set

        if network_type == 'standard':
            pred_class = net(train_X).cpu().data.numpy().argmax(axis=1)
        elif network_type == 'bayesian':
            pred_class = net.predict_by_sampling(train_X, num_samples=N_Samples_Testing).data.cpu().numpy().argmax(axis=1)
        else:
            raise ValueError

        true_class = train_Y.data.cpu().numpy().ravel()

        train_accu = compute_accu(pred_class, true_class, 2)
        print('Epoch', i_ep, '|  Training Accuracy:', train_accu, '%', '| Training Error: ', round(100-train_accu, 2), '%')

        train_accu_lst.append(train_accu)

        # test on testing set

        true_class = test_Y.data.cpu().numpy().ravel()

        if network_type == 'standard':
            pred_class = net(test_X).cpu().data.numpy().argmax(axis=1)
        elif network_type == 'bayesian':
            pred_class = net.predict_by_sampling(test_X, num_samples=N_Samples_Testing).data.cpu().numpy().argmax(axis=1)
        else:
            raise ValueError

        test_accu = compute_accu(pred_class, true_class, 2)
        print('Epoch', i_ep, '|  Test Accuracy:', test_accu, '%', '| Test Error: ', round(100-test_accu, 2), '%')

        test_accu_lst.append(test_accu)

    for p in net.parameters():
        print(p)

    # to report the final test error, I will use the average of test errors of the last 10 epochs
    report_test_accu_mean = np.average(test_accu_lst[-10:])
    report_test_accu_std = np.std(test_accu_lst[-10:])
    report_test_error_mean = round(100-report_test_accu_mean, 2)
    print("Test Accuracy: {}".format(report_test_accu_mean))
    print("Test Error: {}".format(report_test_error_mean))
    print("Test Accuracy/Error Std: {}".format(report_test_accu_std))

    # save result to results folder: using pickle
    result_folder_path = "../../results/{}/".format(experiment_name)
    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)
    with open(result_folder_path+"train_accu_lst.pkl", 'wb') as f:
        pickle.dump(train_accu_lst, f)
    with open(result_folder_path+"test_accu_lst.pkl", 'wb') as f:
        pickle.dump(test_accu_lst, f)
    if save_weights is True:
        torch.save(net, result_folder_path+"weights.pt")
