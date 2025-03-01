import torch
import numpy as np

from helper import *
from models import *


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Prepare MNIST datasets
batch_size = 256
dataset = 'mnist' # choose 'mnist', 'cifar10' 
train_loader, test_loader, chw_in, n_out = load_dataset(dataset, batch_size=batch_size)
seed = None
pretrained = False


# Define the network dimensions
num_steps = 2
n_first_hidden = 100
num_binary_layers = 4
n_hidden = 100
num_hidden_layers = num_binary_layers-1

C_first_hidden = 5
C_hidden = 5
list_C_first_hidden = [i for i in range(1,11)]
list_C_hidden = [i for i in range(1,11)]

list_n_first_hidden = [4*i for i in range(1,11)]
list_num_binary_layers = [i for i in range(3,8)]
list_n_hidden = [4*i for i in range(1,11)]
list_num_steps = [2,4,6,8,12,16,24,32,48,64]
list_num_hidden_layers = [i for i in range(1,7)]


# Training hyperparameters
num_epochs = 20
lr = 1e-3
weight_decay= 5e-4
lr_step = num_epochs/2
lr_gamma = 0.1


# Display hyperparameters
save_path = './example_results/'
os.makedirs(save_path, exist_ok=True)
display_iter = int((60000/batch_size) / 1) #print batch statistics 2 times per epoch
eval_epoch = int(num_epochs / 10) #evaluate params after every 10-th epoch
save_epoch = False #save params at evaluation



# Display parameters
# experiment_snn(train_loader, test_loader, chw_in, n_out, num_steps, n_first_hidden, num_binary_layers, n_hidden, seed=seed,
#             save_path=save_path, pretrained=False, num_epochs=num_epochs, lr=lr, weight_decay=0, lr_step=lr_step, 
#             lr_gamma=lr_gamma, output='spike', display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=False)

# experiment_ann(train_loader, test_loader, n_in, n_out, n_first_hidden, num_hidden_layers, n_hidden, seed=seed,
#                save_path=save_path, pretrained=False, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, 
#                lr_step=lr_step, lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=False)


# experiment_csnn(train_loader, test_loader, chw_in, n_out, num_steps, C_first_hidden, num_binary_layers, C_hidden, 
#             output='spike', seed=None, save_path=save_path, pretrained=False, num_epochs=num_epochs, lr=lr, weight_decay=0, 
#             lr_step=lr_step, lr_gamma=lr_gamma, display_iter =display_iter, eval_epoch=eval_epoch, save_epoch=False)

experiment_cnn(train_loader, test_loader, chw_in, n_out, C_first_hidden, num_hidden_layers, C_hidden, pool=True,
            seed=None, save_path=save_path, pretrained=False, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, 
            lr_step=lr_step, lr_gamma=lr_gamma, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=False)