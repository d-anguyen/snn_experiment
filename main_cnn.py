import torch
import numpy as np

from helper import *
from models import *


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Prepare dataset
batch_size = 512
dataset = 'cifar10' # choose 'mnist', 'cifar10', 'fashion_mnist', 'svhn' , 'stl10'



# Define the network dimensions
num_steps = 8
C_first_hidden = 3
num_binary_layers = 4
C_hidden = 3
num_hidden_layers = num_binary_layers-1

list_C_first_hidden = [i for i in range(1,7)]
list_num_binary_layers = [i for i in range(2,6)]
list_C_hidden = [i for i in range(1,7)]
list_num_steps = [2*i for i in range(1,7)] 
list_num_hidden_layers = [i for i in range(1,4)]



# Training hyperparameters
pretrained = False # set True to only load and evaluate pretrained models, no training needed
num_epochs = 200
lr = 1e-3
weight_decay= 5e-4
lr_step = num_epochs/2
lr_gamma = 0.1


# Display hyperparameters
display_iter = int((60000/batch_size) / 3) #print batch statistics 4 times per epoch
eval_epoch = int(num_epochs / 20) #evaluate and save params after every 20-th epoch
save_epoch = False #choose to save the params at the evaluated epochs or not


# Run different comparisons

compare_csnn(dataset, batch_size=batch_size, num_steps=list_num_steps, C_first_hidden=C_first_hidden, 
                num_binary_layers=num_binary_layers, C_hidden=C_hidden, output='spike',
                num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
                display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './CSNN_comparison/'+dataset+'/', num_trials=5)

compare_csnn(dataset, batch_size=batch_size, num_steps=num_steps, C_first_hidden=list_C_first_hidden, 
                num_binary_layers=num_binary_layers, C_hidden=C_hidden, output='spike',
                num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
                display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './CSNN_comparison/'+dataset+'/', num_trials=5)

compare_csnn(dataset, batch_size=batch_size, num_steps=num_steps, C_first_hidden=C_first_hidden, 
                num_binary_layers=list_num_binary_layers, C_hidden=C_hidden, output='spike',
                num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
                display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './CSNN_comparison/'+dataset+'/', num_trials=5)

compare_csnn(dataset, batch_size=batch_size, num_steps=num_steps, C_first_hidden=C_first_hidden, 
                num_binary_layers=num_binary_layers, n_hidden=list_C_hidden, output='spike',
                num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, lr_gamma=lr_gamma,
                display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './CSNN_comparison/'+dataset+'/', num_trials=5)

compare_cnn(dataset, batch_size=batch_size, C_first_hidden=list_C_first_hidden, num_hidden_layers=num_hidden_layers, 
                C_hidden=C_hidden, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
                lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './CNN_comparison/'+dataset+'/', num_trials=5)

compare_cnn(dataset, batch_size=batch_size, C_first_hidden=C_first_hidden, num_hidden_layers=list_num_hidden_layers, 
                C_hidden=C_hidden, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
                lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './CNN_comparison/'+dataset+'/', num_trials=5)

compare_cnn(dataset, batch_size=batch_size, C_first_hidden=C_first_hidden, num_hidden_layers=num_hidden_layers, 
                C_hidden=list_C_hidden, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
                lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch,
                pretrained=False, save_path = './CNN_comparison/'+dataset+'/', num_trials=5)
