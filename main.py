import torch
import numpy as np

from helper import *
from models import *


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Prepare MNIST datasets
batch_size = 4096
dataset = 'mnist' # choose 'mnist', 'cifar10' 
seed = np.random.randint(300) # later set a seed to fix the initialization
# seed = 30
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
pretrained = False


# Define the network dimensions
num_steps = 8
n_first_hidden = 20
num_binary_layers = 4
n_hidden = 20
num_hidden_layers = num_binary_layers-1

list_n_first_hidden = [4*i for i in range(1,11)] #replace 4 by 20 for cifar10
list_num_binary_layers = [i for i in range(2,8)]
list_n_hidden = [4*i for i in range(1,11)] # increase width for cifar10
list_num_steps = [2,4,6,8,12,16,24,32] #removed 48,64 which last 5,6 hours resp.
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
display_iter = int((60000/batch_size) / 2) #print batch statistics 4 times per epoch
eval_epoch = int(num_epochs / 10) #evaluate and save params after every 10-th epoch
save_epoch = False #choose to save the params at the evaluated epochs or not


# Run different comparisons

# compare_snn(dataset, batch_size=batch_size, num_steps=list_num_steps, n_first_hidden=n_first_hidden, 
#                 num_binary_layers=num_binary_layers, n_hidden=n_hidden, seed = seed, output='spike',
#                 num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
#                 display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
#                 pretrained=False, save_path = './SNN_comparison/'+dataset+'/', num_trials=10)

# compare_snn(dataset, batch_size=batch_size, num_steps=num_steps, n_first_hidden=list_n_first_hidden, 
#                 num_binary_layers=num_binary_layers, n_hidden=n_hidden, seed = seed, output='spike',
#                 num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
#                 display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
#                 pretrained=False, save_path = './SNN_comparison/'+dataset+'/', num_trials=10)

# compare_snn(dataset, batch_size=batch_size, num_steps=num_steps, n_first_hidden=n_first_hidden, 
#                 num_binary_layers=list_num_binary_layers, n_hidden=n_hidden, seed = seed, output='spike',
#                 num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
#                 display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
#                 pretrained=False, save_path = './SNN_comparison/'+dataset+'/', num_trials=10)

# compare_snn(dataset, batch_size=batch_size, num_steps=num_steps, n_first_hidden=n_first_hidden, 
#                 num_binary_layers=num_binary_layers, n_hidden=list_n_hidden, seed = seed, output='spike',
#                 num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
#                 display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
#                 pretrained=False, save_path = './SNN_comparison/'+dataset+'/', num_trials=10)

# compare_ann(dataset, batch_size=batch_size, n_first_hidden=list_n_first_hidden, num_hidden_layers=num_hidden_layers, 
#                 n_hidden=n_hidden, seed = seed, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
#                 lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
#                 pretrained=False, save_path = './ANN_comparison/'+dataset+'/', num_trials=10)

# compare_ann(dataset, batch_size=batch_size, n_first_hidden=n_first_hidden, num_hidden_layers=list_num_hidden_layers, 
#                 n_hidden=n_hidden, seed = seed, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
#                 lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
#                 pretrained=False, save_path = './ANN_comparison/'+dataset+'/', num_trials=10)

# compare_ann(dataset, batch_size=batch_size, n_first_hidden=n_first_hidden, num_hidden_layers=num_hidden_layers, 
#                 n_hidden=list_n_hidden, seed = seed, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
#                 lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
#                 pretrained=False, save_path = './ANN_comparison/'+dataset+'/', num_trials=10)