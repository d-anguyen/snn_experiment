import os
import time
import matplotlib.pyplot as plt
import torch
import models
import train
import numpy as np

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# run the training for one specific hyperparameter set and return statistics after training
def train_and_save(num_steps, n_first_hidden, num_binary_layers, n_hidden, 
               train_loader, test_loader, save_path, pretrained=False, num_epochs=10):    
    # Create a folder to save the results
    name = 'T='+ str(num_steps)+'_784-' + str(n_first_hidden)
    for i in range(num_binary_layers-2):
        name += '-' +str(n_hidden)
    name += '-10/'
    save_path+= name
    os.makedirs(save_path, exist_ok=True)
    
    
    # Randomize the network and train (if no pretrained model is available)
    net = models.SNN(num_steps=num_steps, n_first_hidden=n_first_hidden, 
                     num_binary_layers = num_binary_layers, n_hidden = n_hidden).to(device)
    f = open(save_path+'results.txt', 'w')
    train.print_and_save('Network architecture: '+name, f)
    f.close()
    #print(f'Number of time steps: T={num_steps}')
    print(net)

    if pretrained==True:    
        net.load_state_dict(torch.load(save_path+'params.pth'))
    else:
        start_time = time.time()
        train.train_snn(net, train_loader, test_loader, num_epochs = num_epochs, output='spike')
        train_time = time.time()-start_time
        torch.save(net.state_dict(), save_path+'params.pth')
    
    # Analyze the trained network
    
    file = open(save_path+'results.txt', 'a')
    train_loss, train_acc = train.print_snn_statistics(net, 
                            train_loader, file=file,epoch=num_epochs, train=True, output='spike')
    start_time = time.time()
    test_loss, test_acc = train.print_snn_statistics(net, 
                            test_loader, file=file,epoch=num_epochs, train=False, output='spike')
    test_inference_time = time.time() - start_time
    train.print_and_save(f'Training time: {train_time:.2f} seconds. Test inference time: {test_inference_time:.2f}', file)
    file.close()
    return train_loss, train_acc, test_loss, test_acc, train_time, test_inference_time

# Given a nested list of train/test loss/accuracy and training time and a list of 
# traversed value (with description), create and save a comparison plot 
def plot_comparison(trial_results, list_x, desc,save_path):
    
    avg_results = np.mean(trial_results, axis=0)
    std_devs = np.sqrt(np.var(trial_results, axis=0))
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list_x, avg_results[:,0], label='Train', color='blue')
    plt.plot(list_x, avg_results[:,2], label='Test', color='orange')
    
    # Plot the variance as a shaded region
    plt.fill_between(list_x , 
                 avg_results[:,0] - std_devs[:,0], 
                 avg_results[:,0] + std_devs[:,0], 
                 color='blue', alpha=0.2)
    
    plt.fill_between(list_x , 
                 avg_results[:,2] - std_devs[:,2], 
                 avg_results[:,2] + std_devs[:,2], 
                 color='orange', alpha=0.2)
    plt.xticks(list_x)
    plt.xlabel(desc)
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train/test loss comparison")
    fig.savefig(save_path+'loss.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list_x, avg_results[:,1], label='Train', color='blue')
    plt.plot(list_x, avg_results[:,3], label='Test', color='orange')
    
    # Plot the variance as a shaded region
    plt.fill_between(list_x , 
                 avg_results[:,1] - std_devs[:,1], 
                 avg_results[:,1] + std_devs[:,1], 
                 color='blue', alpha=0.2)
    
    plt.fill_between(list_x , 
                 avg_results[:,3] - std_devs[:,3], 
                 avg_results[:,3] + std_devs[:,3], 
                 color='orange', alpha=0.2)
    plt.xticks(list_x)
    plt.xlabel(desc)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Train/test accuracy comparison")
    fig.savefig(save_path+'accuracy.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list_x, avg_results[:,4], label='Training time', color='blue')
    plt.plot(list_x, avg_results[:,5], label='Test inference', color='orange')
    
    # Plot the variance as a shaded region
    plt.fill_between(list_x , 
                 avg_results[:,4] - std_devs[:,4], 
                 avg_results[:,4] + std_devs[:,4], 
                 color='blue', alpha=0.2)
    
    plt.fill_between(list_x , 
                 avg_results[:,5] - std_devs[:,5], 
                 avg_results[:,5] + std_devs[:,5], 
                 color='orange', alpha=0.2)
    plt.xticks(list_x)
    plt.xlabel(desc)
    plt.ylabel("Time in seconds")
    plt.legend()
    plt.title("Training and test inference time")
    fig.savefig(save_path+'time.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
def compare(variable, train_loader, test_loader, seed = None, num_epochs = 10, save_path = './mnist_comparison/', num_trials=1):
    # Fix a seed in case we want a single fixed trial 
    if (num_trials==1) and (seed is not None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    trial_results = []
    path = save_path
    for _ in range(num_trials):
        # Define the default hyperparameters
        num_steps = 2
        n_first_hidden = 100
        num_binary_layers = 4
        n_hidden = 100
        list_loss_acc = []
        save_path =path+ 'compare_'+variable+'_trial_'+str(_+1)+'/'
        os.makedirs(save_path, exist_ok=True)
        
        # Run experiments depending on the given concern
        if variable == 'num_steps':
            list_x = [2**i for i in range(1,7)]
            for num_steps in list_x:
                list_loss_acc.append(train_and_save(num_steps, n_first_hidden, num_binary_layers, 
                           n_hidden, train_loader, test_loader, save_path, num_epochs=num_epochs))
            plot_desc = 'Number of time steps'
            
        elif variable == 'n_first_hidden':
            list_x = [20*i for i in range(1,11)] #change to 11
            for n_first_hidden in list_x:
                list_loss_acc.append(train_and_save(num_steps, n_first_hidden, num_binary_layers, 
                           n_hidden, train_loader, test_loader, save_path, num_epochs=num_epochs))
            plot_desc = 'Width of the first hidden layer'
            
        elif variable == 'num_binary_layers':
            list_x = [i for i in range(2,8)]
            for num_binary_layers in list_x:
                list_loss_acc.append(train_and_save(num_steps, n_first_hidden, num_binary_layers, 
                           n_hidden, train_loader, test_loader, save_path, num_epochs=num_epochs))
            plot_desc = 'Number of binary layers'
                    
        elif variable == 'n_hidden':
            list_x = [20*i for i in range(1,11)]
            for n_hidden in list_x:
                list_loss_acc.append(train_and_save(num_steps, n_first_hidden, num_binary_layers, 
                           n_hidden, train_loader, test_loader, save_path, num_epochs=num_epochs))
            plot_desc = 'Width of the subsequent hidden layer'
        else:
            raise ValueError("Choose to vary either 'n_first_hidden', 'num_steps', 'num_binary_layers', or 'n_hidden'.")
        trial_results.append(np.array(list_loss_acc))
        
    trial_results = np.array(trial_results)
    
    plot_comparison(trial_results, list_x, plot_desc, save_path)
    
