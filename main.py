import snntorch as snn
import os
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import matplotlib.pyplot as plt

import models
import train

from helper import *


# Training hyperparameters
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = 'mnist' #'cifar10
num_epochs = 200
batch_size = 2048
lr= 1e-3 
weight_decay=0 
lr_step=100 
lr_gamma=0.1

# Display parameters
eval_epoch = num_epochs/10 #evaluate the network after this number of epochs
display_iter = int( (60000/batch_size) / 4 ) #display batch statistics 4 times every epoch
# turn to None to do silent training


# Prepare MNIST/CIFAR10 datasets
train_loader, test_loader = load_dataset(dataset, batch_size=batch_size,shuffle=True)



# Comparison of SNNs hyperparameters. Choose either 'n_first_hidden', 'num_steps', 'num_binary_layers' or 'n_hidden'

#compare_snn('n_first_hidden', train_loader, test_loader, num_epochs=20, num_trials=10)
#compare_snn('n_hidden', train_loader, test_loader, num_epochs=200, num_trials=1)
#compare_snn('num_steps', train_loader, test_loader, num_epochs=20, num_trials=10)
#compare_snn('num_binary_layers', train_loader, test_loader, num_epochs=20, num_trials=10)


# Comparison of ANNs hyperparameters. Choose either 'n_first_hidden', 'num_hidden_layers' or 'n_hidden'

#compare_ann('n_first_hidden', train_loader, test_loader, num_epochs=20, num_trials=10, weight_decay=1e-4)
#compare_ann('n_hidden', train_loader, test_loader, num_epochs=20, num_trials=10, weight_decay=1e-4)
#compare_ann('num_hidden_layers', train_loader, test_loader, num_epochs=20, num_trials=10, weight_decay=1e-4)

