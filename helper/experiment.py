import os
import time
import pickle

import torch
import numpy as np
import datetime


from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import *
from train import *
from helper.plot import *




dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_dataset(name, batch_size=256, shuffle=True):
    data_path = '/tmp/data/'+name  
    if name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        chw_in, n_out = (1,28,28), 10

    elif name == 'fashion_mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        chw_in, n_out = (1,28,28), 10
    elif name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        chw_in, n_out = (3,32,32) ,10
    elif name == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])])
        train_set = datasets.SVHN(data_path, split='train', download=True, transform=transform)
        test_set = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        chw_in, n_out = (3,32,32), 10
    elif name == 'stl10':
        transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize([0.4467, 0.4398, 0.4066], [0.2240, 0.2215, 0.2239])])
        train_set = datasets.STL10(data_path, split='train', download=True, transform=transform)
        test_set = datasets.STL10(data_path, split='test', download=True, transform=transform)
        chw_in, n_out = (3,96,96), 10
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    return train_loader, test_loader, chw_in, n_out 



def print_and_save(text_str, file):
    print(text_str)
    if file is not None:
        print(text_str, file=file)



#################################################################
# run the training for one specific hyperparameter set and return statistics after several epochs
def experiment_snn(train_loader, test_loader, chw_in, n_out, num_steps, n_first_hidden, num_binary_layers, n_hidden, 
            output='spike', seed=None, save_path=None, pretrained=False, num_epochs=200, lr=1e-3, weight_decay=0, 
            lr_step=50, lr_gamma=0.1, display_iter =None, eval_epoch=None, save_epoch=False):    
    saved_args = {**locals()}
    [saved_args.pop(key) for key in ['train_loader', 'test_loader']]
    n_in = chw_in[0]*chw_in[1]*chw_in[2]
    if save_path is not None:
        # Create a folder to save the results
        name = 'T='+ str(num_steps)+'_' + str(n_in) +'-' + str(n_first_hidden)
        for i in range(num_binary_layers-2):
            name += '-' +str(n_hidden)
        name += '-'+str(n_out)+'/'
        save_path+= name
        os.makedirs(save_path, exist_ok=True)
        
        file = open(save_path+'results.txt', 'w')
        print_and_save(saved_args, file)
        print_and_save('Network architecture: '+name, file)
        file.close()
    
    # Randomize the network and train (if no pretrained model is available)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    net = SNN(n_in=n_in, n_out=n_out, num_steps=num_steps, n_first_hidden=n_first_hidden, 
                     num_binary_layers = num_binary_layers, n_hidden = n_hidden).to(device)
    print(net)

    if pretrained==True:    
        net.load_state_dict(torch.load(save_path+'trained_params.pth'))
        train_time = None
    else:
        start_time = time.time()
        train_hist, test_hist, batch_hist = train_snn(net, train_loader, test_loader, output='spike',  
                  num_epochs=num_epochs, lr= lr, weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma,
                  display_iter= display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch, save_path=save_path)
        
        train_time = time.time()-start_time
        if save_path is not None:
            torch.save(net.state_dict(), save_path+'trained_params.pth')
    
    # Evaluate the trained network
    print('\n ############################################')
    train_loss, train_acc = evaluate_snn(net, train_loader)
    
    start_time = time.time()
    test_loss, test_acc = evaluate_snn(net,test_loader)
    test_inference_time = time.time() - start_time
    
    file = None
    if save_path is not None:
        file = open(save_path+'results.txt', 'a')
    
    print_and_save(f'Train loss: {train_loss:.2f}, train accuracy: {train_acc*100:.2f}%', file)
    print_and_save(f'Test loss: {test_loss:.2f}, test accuracy: {test_acc*100:.2f}%', file)
    if pretrained == False:
        print_and_save(f'Training time: {str(datetime.timedelta(seconds= int(train_time)) )} seconds', file)
    else:
        print_and_save('Training time: unknown', file)
    print_and_save(f'Test inference time: {test_inference_time:.2f} seconds', file)
    
    if file is not None:
        file.close()
    plot_learning_curve(train_hist, test_hist, batch_hist, plot_batch= False, 
                    eval_epoch=eval_epoch, num_epochs=num_epochs, desc = '', save_path = save_path)
    return train_loss, train_acc, test_loss, test_acc, train_time, test_inference_time


def experiment_csnn(train_loader, test_loader, chw_in, n_out, num_steps, C_first_hidden, num_binary_layers, C_hidden, 
            output='spike', seed=None, save_path=None, pretrained=False, num_epochs=200, lr=1e-3, weight_decay=0, 
            lr_step=50, lr_gamma=0.1, display_iter =None, eval_epoch=None, save_epoch=False):    
    saved_args = {**locals()}
    [saved_args.pop(key) for key in ['train_loader', 'test_loader']]
    if save_path is not None:
        # Create a folder to save the results
        name = 'T='+ str(num_steps)+'_C' + str(chw_in[0]) +'-C' + str(C_first_hidden)
        for i in range(num_binary_layers-2):
            name += '-C' +str(C_hidden)
        name += '-'+str(n_out)+'/'
        save_path+= name
        os.makedirs(save_path, exist_ok=True)
        
        file = open(save_path+'results.txt', 'w')
        print_and_save('Network architecture: '+name, file)
        print_and_save(saved_args, file)
        file.close()
    
    # Randomize the network and train (if no pretrained model is available)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    net = CSNN(chw_in=chw_in, n_out=n_out, num_steps=num_steps, C_first_hidden=C_first_hidden, 
                     num_binary_layers = num_binary_layers, C_hidden = C_hidden).to(device)
    print(net)

    if pretrained==True:    
        net.load_state_dict(torch.load(save_path+'trained_params.pth'))
        train_time = None
    else:
        start_time = time.time()
        train_hist, test_hist, batch_hist = train_snn(net, train_loader, test_loader, output='spike',  
                  num_epochs=num_epochs, lr= lr, weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma,
                  display_iter= display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch, save_path=save_path)
        
        train_time = time.time()-start_time
        if save_path is not None:
            torch.save(net.state_dict(), save_path+'trained_params.pth')
    
    # Evaluate the trained network
    print('\n ############################################')
    train_loss, train_acc = evaluate_snn(net, train_loader)
    
    start_time = time.time()
    test_loss, test_acc = evaluate_snn(net,test_loader)
    test_inference_time = time.time() - start_time
    
    file = None
    if save_path is not None:
        file = open(save_path+'results.txt', 'a')
    
    print_and_save(f'Train loss: {train_loss:.2f}, train accuracy: {train_acc*100:.2f}%', file)
    print_and_save(f'Test loss: {test_loss:.2f}, test accuracy: {test_acc*100:.2f}%', file)
    if pretrained == False:
        print_and_save(f'Training time: {str(datetime.timedelta(seconds= int(train_time)) )} seconds', file)
    else:
        print_and_save('Training time: unknown', file)
    print_and_save(f'Test inference time: {test_inference_time:.2f} seconds', file)
    
    if file is not None:
        file.close()
    plot_learning_curve(train_hist, test_hist, batch_hist, plot_batch= False, 
                    eval_epoch=eval_epoch, num_epochs=num_epochs, desc = '', save_path = save_path)
    return train_loss, train_acc, test_loss, test_acc, train_time, test_inference_time

def experiment_cnn(train_loader, test_loader, chw_in, n_out, C_first_hidden, num_hidden_layers, C_hidden, pool=False,
            seed=None, save_path=None, pretrained=False, num_epochs=200, lr=1e-3, weight_decay=5e-4, 
            lr_step=50, lr_gamma=0.1, display_iter =None, eval_epoch=None, save_epoch=False):    
    saved_args = {**locals()}
    [saved_args.pop(key) for key in ['train_loader', 'test_loader']]
    n_in = chw_in[0]*chw_in[1]*chw_in[2]
    # Create a folder to save the results
    name = 'ANN_'+str(chw_in[0])+ '-C' + str(C_first_hidden)
    if pool:
        name+='-P'
    for i in range(num_hidden_layers-1):
        name += '-C' +str(C_hidden)
    if pool:
        name+='-P'
    name += '-' + str(n_out)+'/'
    save_path+= name
    os.makedirs(save_path, exist_ok=True)
    
    # Randomize the network and train (if no pretrained model is available)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    net = CNN(chw_in=chw_in, n_out=n_out, C_first_hidden=C_first_hidden, num_hidden_layers = num_hidden_layers, 
              C_hidden = C_hidden, pool=pool).to(device)
    file = open(save_path+'results.txt', 'w')
    print_and_save(saved_args, file)
    print_and_save('Network architecture: '+name, file)
    file.close()
    
    print(net)

    if pretrained==True:    
        net.load_state_dict(torch.load(save_path+'trained_params.pth'))
        train_time = None
    else:
        start_time = time.time()
        train_hist, test_hist, batch_hist = train_ann(net, train_loader, test_loader, num_epochs=num_epochs, lr= lr, 
                  weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma, 
                  display_iter= display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch, save_path=save_path)
        train_time = time.time()-start_time
        if save_path is not None:
            torch.save(net.state_dict(), save_path+'trained_params.pth')
    
    # Evaluate the trained network
    train_loss, train_acc = evaluate_ann(net, train_loader)
    
    start_time = time.time()
    test_loss, test_acc = evaluate_ann(net,test_loader)
    test_inference_time = time.time() - start_time
    
    print('\n ############################################')
    file = None
    if save_path is not None:
        file = open(save_path+'results.txt', 'a')
    
    print_and_save(f'Train loss: {train_loss:.2f}, train accuracy: {train_acc*100:.2f}%', file)
    print_and_save(f'Test loss: {test_loss:.2f}, test accuracy: {test_acc*100:.2f}%', file)
    if pretrained == False:
        print_and_save(f'Training time: {str(datetime.timedelta( seconds= int(train_time) ))} seconds', file)
    else:
        print_and_save('Training time: unknown', file)
    print_and_save(f'Test inference time: {test_inference_time:.2f} seconds', file)
    
    if file is not None:
        file.close()
    plot_learning_curve(train_hist, test_hist, batch_hist, plot_batch= False, 
                    eval_epoch=eval_epoch, num_epochs=num_epochs, desc = '', save_path = save_path)
    
    return train_loss, train_acc, test_loss, test_acc, train_time, test_inference_time



def experiment_ann(train_loader, test_loader, chw_in, n_out, n_first_hidden, num_hidden_layers, n_hidden, 
            seed=None, save_path=None, pretrained=False, num_epochs=200, lr=1e-3, weight_decay=5e-4, 
            lr_step=50, lr_gamma=0.1, display_iter =None, eval_epoch=None, save_epoch=False):    
    saved_args = {**locals()}
    [saved_args.pop(key) for key in ['train_loader', 'test_loader']]
    n_in = chw_in[0]*chw_in[1]*chw_in[2]
    # Create a folder to save the results
    name = 'ANN_'+str(n_in)+ '-' + str(n_first_hidden)
    for i in range(num_hidden_layers-1):
        name += '-' +str(n_hidden)
    name += '-' + str(n_out)+'/'
    save_path+= name
    os.makedirs(save_path, exist_ok=True)
    
    # Randomize the network and train (if no pretrained model is available)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    net = ANN(n_in=n_in, n_out=n_out, n_first_hidden=n_first_hidden, num_hidden_layers = num_hidden_layers, 
              n_hidden = n_hidden).to(device)
    file = open(save_path+'results.txt', 'w')
    print_and_save(saved_args, file)
    print_and_save('Network architecture: '+name, file)
    file.close()
    
    print(net)

    if pretrained==True:    
        net.load_state_dict(torch.load(save_path+'trained_params.pth'))
        train_time = None
    else:
        start_time = time.time()
        train_hist, test_hist, batch_hist = train_ann(net, train_loader, test_loader, num_epochs=num_epochs, lr= lr, 
                  weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma, 
                  display_iter= display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch, save_path=save_path)
        train_time = time.time()-start_time
        if save_path is not None:
            torch.save(net.state_dict(), save_path+'trained_params.pth')
    
    # Evaluate the trained network
    train_loss, train_acc = evaluate_ann(net, train_loader)
    
    start_time = time.time()
    test_loss, test_acc = evaluate_ann(net,test_loader)
    test_inference_time = time.time() - start_time
    
    print('\n ############################################')
    file = None
    if save_path is not None:
        file = open(save_path+'results.txt', 'a')
    
    print_and_save(f'Train loss: {train_loss:.2f}, train accuracy: {train_acc*100:.2f}%', file)
    print_and_save(f'Test loss: {test_loss:.2f}, test accuracy: {test_acc*100:.2f}%', file)
    if pretrained == False:
        print_and_save(f'Training time: {str(datetime.timedelta( seconds= int(train_time) ))} seconds', file)
    else:
        print_and_save('Training time: unknown', file)
    print_and_save(f'Test inference time: {test_inference_time:.2f} seconds', file)
    
    if file is not None:
        file.close()
    plot_learning_curve(train_hist, test_hist, batch_hist, plot_batch= False, 
                    eval_epoch=eval_epoch, num_epochs=num_epochs, desc = '', save_path = save_path)
    
    return train_loss, train_acc, test_loss, test_acc, train_time, test_inference_time





#####################################################################################
# One of the network hyperparameters will be varied from elements of a given list. 
# Others will be given fixed. Run each experiment with that set of network hyperparams
# a number of times and output the results. 
def compare_snn(dataset, num_steps, n_first_hidden, num_binary_layers, n_hidden, output='spike', batch_size=256, 
                num_epochs=200, lr=1e-3, weight_decay=0, lr_step=100, lr_gamma=0.1, display_iter =None, 
                eval_epoch=None, save_epoch=False, pretrained=False, save_path = './SNN_comparison/', num_trials=1):
    
    saved_args = {**locals()}
    [saved_args.pop(key) for key in ['train_loader', 'test_loader']]
    
    file = open(save_path+'hyperparams.txt', 'w')
    print_and_save(saved_args, file)
    file.close()
    
    train_loader, test_loader, n_in, n_out = load_dataset(dataset, batch_size= batch_size)
    trial_results = []
    path = save_path
    
    if isinstance(num_steps, list):
        for trial in range(num_trials):
            # Create a list to save loss/accuracy of each experiment
            list_loss_acc = []
            save_path =path+ 'compare_num_steps/trial_'+str(trial+1)+'/'
            os.makedirs(save_path, exist_ok=True)
            print(f'###### Experiments on number of time steps ###### Trial number {trial+1} ######')
            for val in num_steps:
                list_loss_acc.append(
                experiment_snn(train_loader, test_loader, n_in=n_in, n_out=n_out, num_steps=val, n_first_hidden=n_first_hidden, 
                num_binary_layers=num_binary_layers, n_hidden=n_hidden, save_path=save_path, pretrained=False, 
                num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma, 
                output=output, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch) )           
                trial_results.append(list_loss_acc)
        
        plot_comparison(trial_results, num_steps, 'Number of time steps', path+'compare_num_steps/')
        with open(path + 'results.pkl', 'wb') as f:
            pickle.dump(trial_results, f)
            
    elif isinstance(n_first_hidden, list):
        for trial in range(num_trials):
            # Create a list to save loss/accuracy of each experiment
            list_loss_acc = []
            save_path =path+ 'compare_n_first_hidden/trial_'+str(trial+1)+'/'
            os.makedirs(save_path, exist_ok=True)
            print(f'###### Experiments on width of first hidden layer ###### Trial number {trial+1} ######')
            for val in n_first_hidden:
                list_loss_acc.append(
                experiment_snn(train_loader, test_loader, n_in=n_in, n_out=n_out, num_steps=num_steps, n_first_hidden=val, 
                num_binary_layers=num_binary_layers, n_hidden=n_hidden, save_path=save_path, pretrained=False, 
                num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma, 
                output=output, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch) )
            trial_results.append(list_loss_acc)
                
        plot_comparison(trial_results, n_first_hidden, 'Width of the first hidden layers', path+'compare_n_first_hidden/')
        with open(path + 'results.pkl', 'wb') as f:
            pickle.dump(trial_results, f)
            
    elif isinstance(num_binary_layers, list):
        for trial in range(num_trials):
            # Create a list to save loss/accuracy of each experiment
            list_loss_acc = []
            save_path =path+ 'compare_num_binary_layers/trial_'+str(trial+1)+'/'
            os.makedirs(save_path, exist_ok=True)
            print(f'###### Experiments on number of binary layers ###### Trial number {trial+1} ######')
            for val in num_binary_layers:
                list_loss_acc.append(
                experiment_snn(train_loader, test_loader, n_in=n_in, n_out=n_out, num_steps=num_steps, n_first_hidden=n_first_hidden, 
                num_binary_layers=val, n_hidden=n_hidden, save_path=save_path, pretrained=False, 
                num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma, 
                output=output, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch) )
            trial_results.append(list_loss_acc)
        plot_comparison(trial_results, num_binary_layers, 'Number of binary layers', path+'compare_num_binary_layers/')
        with open(path + 'results.pkl', 'wb') as f:
            pickle.dump(trial_results, f)
            
    elif isinstance(n_hidden, list):
        for trial in range(num_trials):
            # Create a list to save loss/accuracy of each experiment
            list_loss_acc = []
            save_path =path+ 'compare_n_hidden/trial_'+str(trial+1)+'/'
            os.makedirs(save_path, exist_ok=True)
            print(f'###### Experiments on width of subsequent hidden layers ###### Trial number {trial+1} ######')
            for val in n_hidden:
                list_loss_acc.append(
                experiment_snn(train_loader, test_loader, n_in=n_in, n_out=n_out, num_steps=num_steps, n_first_hidden=n_first_hidden, 
                num_binary_layers=num_binary_layers, n_hidden=val, save_path=save_path, pretrained=False, 
                num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, lr_gamma=lr_gamma, 
                output=output, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch) )
            trial_results.append(list_loss_acc)
        plot_comparison(trial_results, n_hidden, 'Width of subsequent hidden layers', path+'compare_n_hidden/')
        with open(path + 'results.pkl', 'wb') as f:
            pickle.dump(trial_results, f)
            
    else:
        raise ValueError("Choose to vary only 1 hyperparameter, either 'n_first_hidden', 'num_steps', 'num_binary_layers', or 'n_hidden'.")    
    
    

#############################################################
def compare_ann(dataset, n_first_hidden, num_hidden_layers, n_hidden, batch_size=256, num_epochs=200, lr=1e-3, 
                weight_decay=0, lr_step=100, lr_gamma=0.1, display_iter =None, eval_epoch=None, save_epoch=False,
                pretrained=False, save_path = './ANN_comparison/', num_trials=1):
    
    saved_args = {**locals()}
    [saved_args.pop(key) for key in ['train_loader', 'test_loader']]
    
    file = open(save_path+'hyperparams.txt', 'w')
    print_and_save(saved_args, file)
    file.close()
    
    train_loader, test_loader, n_in, n_out = load_dataset(dataset, batch_size= batch_size) 
    trial_results = []
    path = save_path
    

    if isinstance(n_first_hidden, list):
        for trial in range(num_trials):
            # Create a list to save loss/accuracy of each experiment
            list_loss_acc = []
            save_path =path+ 'compare_n_first_hidden/trial'+str(trial+1)+'/'
            os.makedirs(save_path, exist_ok=True)
            print(f'###### Experiments on width of first hidden layer ###### Trial number {trial+1} ######')
            for val in n_first_hidden:
                list_loss_acc.append(
                experiment_ann(train_loader, test_loader, n_in = n_in, n_out=n_out, n_first_hidden=val, num_hidden_layers=num_hidden_layers,
                n_hidden=n_hidden, save_path=save_path, pretrained=False, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
                lr_gamma=lr_gamma, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch) )
            trial_results.append(list_loss_acc)
        plot_comparison(trial_results, n_first_hidden, 'Width of the first hidden layers', path+'compare_n_first_hidden/')
        with open(path + 'results.pkl', 'wb') as f:
            pickle.dump(trial_results, f)
            
    elif isinstance(num_hidden_layers, list):
        for trial in range(num_trials):
            # Create a list to save loss/accuracy of each experiment
            list_loss_acc = []
            save_path =path+ 'compare_num_hidden_layers/trial_'+str(trial+1)+'/'
            os.makedirs(save_path, exist_ok=True)
            print(f'###### Experiments on number of hidden layers ###### Trial number {trial+1} ######')
            for val in num_hidden_layers:
                list_loss_acc.append(
                experiment_ann(train_loader, test_loader, n_in=n_in, n_out=n_out, n_first_hidden=n_first_hidden, num_hidden_layers=val, 
                n_hidden=n_hidden, save_path=save_path, pretrained=False, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, 
                lr_step=lr_step, lr_gamma=lr_gamma, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch) )
            trial_results.append(list_loss_acc)
        plot_comparison(trial_results, num_binary_layers, 'Number of hidden layers', path+'compare_num_binary_layers/')
        with open(path + 'results.pkl', 'wb') as f:
            pickle.dump(trial_results, f)
            
    elif isinstance(n_hidden, list):
        for trial in range(num_trials):
            # Create a list to save loss/accuracy of each experiment
            list_loss_acc = []
            save_path =path+ 'compare_n_hidden/trial_'+str(trial+1)+'/'
            os.makedirs(save_path, exist_ok=True)
            print(f'###### Experiments on width of subsequent hidden layers ###### Trial number {trial+1} ######')
            for val in n_hidden:
                list_loss_acc.append(
                experiment_ann(train_loader, test_loader, n_in=n_in, n_out=n_out, n_first_hidden=n_first_hidden, num_hidden_layers=num_hidden_layers, 
                n_hidden=val, save_path=save_path, pretrained=False, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, lr_step=lr_step, 
                lr_gamma=lr_gamma, display_iter=display_iter, eval_epoch=eval_epoch, save_epoch=save_epoch) )
            trial_results.append(list_loss_acc)
        plot_comparison(trial_results, n_hidden, 'Width of subsequent hidden layers', path+'compare_n_hidden/')
        with open(path + 'results.pkl', 'wb') as f:
            pickle.dump(trial_results, f)
            
    else:
        raise ValueError("Choose only one hyperparameter to change, either 'n_first_hidden', 'num_hidden_layers', or 'n_hidden'.")    
    

