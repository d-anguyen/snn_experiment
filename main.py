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


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Prepare MNIST datasets
data_path='/tmp/data/mnist'
batch_size = 256
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


# Comparison of SNNs hyperparameters. Choose either 'n_first_hidden', 'num_steps', 'num_binary_layers' or 'n_hidden'

#compare_snn('n_first_hidden', train_loader, test_loader, num_epochs=20, num_trials=10)
#compare_snn('n_hidden', train_loader, test_loader, num_epochs=20, num_trials=10)
#compare_snn('num_steps', train_loader, test_loader, num_epochs=20, num_trials=10)
#compare_snn('num_binary_layers', train_loader, test_loader, num_epochs=20, num_trials=10)


# Comparison of ANNs hyperparameters. Choose either 'n_first_hidden', 'num_hidden_layers' or 'n_hidden'

#compare_ann('n_first_hidden', train_loader, test_loader, num_epochs=20, num_trials=10, weight_decay=1e-4)
#compare_ann('n_hidden', train_loader, test_loader, num_epochs=20, num_trials=10, weight_decay=1e-4)
compare_ann('num_hidden_layers', train_loader, test_loader, num_epochs=20, num_trials=10, weight_decay=1e-4)

