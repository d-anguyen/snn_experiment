import torch
import numpy as np

from helper import *
from models import *


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Prepare MNIST datasets
batch_size = 2048
dataset = 'cifar10' # choose 'mnist', 'cifar10' 
seed = np.random.randint(300) # later set a seed to fix the initialization
# seed = 30
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
pretrained = False


# Define the network dimensions
num_steps = 8
n_first_hidden = 100
num_binary_layers = 4
n_hidden = 100
num_hidden_layers = num_binary_layers-1

list_n_first_hidden = [20*i for i in range(1,11)]
list_num_binary_layers = [i for i in range(2,8)]
list_n_hidden = [20*i for i in range(1,11)]
list_num_steps = [2,4,6,8,12,16,24,32,48,64]
list_num_hidden_layers = [i for i in range(1,7)]


# Training hyperparameters
num_epochs = 200
lr = 1e-3
weight_decay= 5e-4
lr_step = num_epochs/2
lr_gamma = 0.1


# Display hyperparameters
#save_path = './example_results/'
#os.makedirs(save_path, exist_ok=True)
display_iter = int((60000/batch_size) / 4) #print batch statistics 4 times per epoch
eval_epoch = int(num_epochs / 10) #evaluate and save params after every 10-th epoch
save_epoch = True



# Display parameters
# experiment_snn(num_steps, n_first_hidden, num_binary_layers, n_hidden, 
#                train_loader, test_loader, save_path=save_path, pretrained=False, 
#                num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
#                output='spike', display_iter =50, eval_epoch=2, save_epoch=True)


# experiment_ann(n_first_hidden, num_hidden_layers, n_hidden, 
#                train_loader, test_loader, save_path=save_path, pretrained=False, 
#                num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma,
#                display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=True)

#compare_snn(train_loader, test_loader, num_steps=2, n_first_hidden=[3,4], num_binary_layers=4, n_hidden=30)


compare_snn(dataset, batch_size=batch_size, num_steps=list_num_steps, n_first_hidden=n_first_hidden, 
                num_binary_layers=num_binary_layers, n_hidden=n_hidden, seed = seed, output='spike',
                num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
                display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './snn_comparison/'+dataset+'/', num_trials=10)


