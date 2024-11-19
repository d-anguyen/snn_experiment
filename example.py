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


# Choose either 'num_first_hidden', 'num_steps', 'num_binary_layers' or 'n_hidden'
compare('n_first_hidden', train_loader, test_loader, seed = 30, num_epochs=20, num_trials=10)



# # Define the hyperparameters
# num_steps = 2
# n_first_hidden = 30
# num_binary_layers = 3
# n_hidden = 20

# seed = np.random.randint(100) # later set a seed to fix the initialization
# # seed = 30
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# pretrained = False

# # Create a folder to save results
# save_path = './example_results/'

# name = 'T='+ str(num_steps)+'_784-' + str(n_first_hidden)
# for i in range(num_binary_layers):
#     name += '-' +str(n_hidden)
# name += '-10/'
# save_path+= name
# os.makedirs(save_path, exist_ok=True)
# # Create a file to save accuracy
# if not os.path.exists(save_path+'results.txt'):
#     with open(save_path+'results.txt', 'w'): pass

# net = models.SNN(num_steps=num_steps, n_first_hidden=n_first_hidden, 
#                  num_binary_layers = num_binary_layers, n_hidden = n_hidden).to(device)
# print(f'Number of time steps: T={num_steps}')
# print(net)

# if pretrained==True:    
#     net.load_state_dict(torch.load(save_path+'params.pth'))
# else:
#     train.train_snn(net, train_loader, test_loader, num_epochs = 10, output='spike')

# file = open(save_path+'results.txt', 'a')
# train.print_snn_statistics(net, train_loader, epoch=10, file=file, train=True, output='spike')
# train.print_snn_statistics(net, test_loader, epoch=10, file=file, train=False, output='spike')
# file.close()















