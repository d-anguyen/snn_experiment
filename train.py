import torch
import snntorch.functional as SF
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import os
import numpy as np



dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")



# Let's set output to be 'spike', we may think about mem_output later. 
def train_snn(net, train_loader, test_loader, num_epochs=100, 
              output='spike', lr= 1e-3, weight_decay=0, lr_step=50, lr_gamma=0.1,
              display_iter= None, eval_epoch=None, save_epoch=False, save_path=None):
    num_steps = net.num_steps # only used for membrane potential output
    if output=='mem':
        loss = nn.CrossEntropyLoss() # later take sum_t CE(mem_out(t), label)
    elif output=='spike':
        loss = SF.ce_count_loss() #CE(ave_t(spk_out(t)), label)
        #loss = SF.ce_rate_loss() #average over time of CE(spk_out(t), label)
    if save_epoch:
        os.makedirs(save_path+'params/', exist_ok = True)
            
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    
    # To display statistics
    train_hist, test_hist, batch_hist = [], [], [] #should be [[loss,accuracy] for each epoch/batch]
    
    
    # Outer training loop        
    for epoch in tqdm(range(num_epochs), desc='Training epoch'):
        print(f"#### Learning rate {scheduler.get_last_lr()[0]:.2e} ####")
        # Minibatch training loop
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
    
            # forward pass
            net.train()
            spk_rec, mem_rec = net(data)
            
            if output=='mem':
            # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], targets)
            else: 
            #output=='spike'
                loss_val = loss(spk_rec, targets)
    
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        
            # Show statistics within the minibatch
            if (display_iter is not None) and (i%display_iter==0):
                net.eval()
                acc = SF.accuracy_rate(spk_rec, targets)
                batch_hist.append([loss_val.item(), acc])
                print(f"--- Iteration {i} --- Train Loss: {loss_val.item():.2f} --- Minibatch accuracy: {acc * 100:.2f}%")
        
        scheduler.step()     
        print('\n')
        
        
        # Evaluation
        if eval_epoch is not None:
            # Evaluate after several epochs (results are printed out but not saved)
            if epoch%eval_epoch==0 or epoch == num_epochs-1:
                train_loss, train_acc = evaluate_snn(net, train_loader, output=output)
                test_loss, test_acc = evaluate_snn(net, test_loader, output=output)
                print(f'Statistics at epoch {epoch+1}:')
                print(f'-Train loss: {train_loss:.2f}, train accuracy: {train_acc*100:.2f}%')
                print(f'-Test loss: {test_loss:.2f}, test accuracy: {test_acc*100:.2f}%')
                
                train_hist.append([train_loss, train_acc])
                test_hist.append([test_loss, test_acc])
                
                # Save the network parameters if needed
                if (save_path is not None) and save_epoch:
                    torch.save(net.state_dict(), save_path+'params/params_after_epoch_' + str(epoch+1)+'.pth')
                    
            # Evaluation in the end of training (and saving results) is moved to other function
            # so if we set eval_epoch to None, the training is quite plain in the sense that 
            # we mostly don't do evaluation.
    
    
    # Return the dataset/batch statistics to plot learning curve
    if eval_epoch is not None:           
        return train_hist, test_hist, batch_hist
    else: 
        return None, None, batch_hist

def train_ann(net, train_loader, test_loader, num_epochs=100, lr= 1e-3, weight_decay=0, lr_step=50, 
              lr_gamma=0.1, display_iter= None, eval_epoch=None, save_epoch=False, save_path=None):
    loss = nn.CrossEntropyLoss()    

    if save_epoch:
        os.makedirs(save_path+'params', exist_ok = True)
                
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    # To display statistics
    train_hist, test_hist, batch_hist = [], [], [] #should be [[loss,accuracy] for each epoch/batch]
    # Outer training loop        
    for epoch in tqdm(range(num_epochs), desc='Training epoch'):
        print(f"#### Learning rate {scheduler.get_last_lr()[0]:.2e} ####")
        # Minibatch training loop
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
    
            # forward pass
            net.train()
            out = net(data)
            loss_val = loss(out, targets)
    
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # Show statistics within the minibatch
            if (display_iter is not None) and (i%display_iter==0):
                net.eval()
                _, predicted = torch.max(out.data, 1)
                acc = (predicted == targets).sum().item()/targets.size(0)
                batch_hist.append([loss_val.item(), acc])
                print(f"--- Iteration {i} --- Train Loss: {loss_val.item():.2f} --- Minibatch accuracy: {acc * 100:.2f}%")
        
        scheduler.step()     
        print('\n')
        
        # Evaluation
        if eval_epoch is not None:
            # Evaluate after several epochs (results are printed out but not saved)
            if epoch%eval_epoch==0 or epoch == num_epochs-1:
                train_loss, train_acc = evaluate_ann(net, train_loader)
                test_loss, test_acc = evaluate_ann(net, test_loader)
                print(f'Statistics at epoch {epoch+1}:')
                print(f'-Train loss: {train_loss:.2f}, train accuracy: {train_acc*100:.2f}%')
                print(f'-Test loss: {test_loss:.2f}, test accuracy: {test_acc*100:.2f}%')
                
                train_hist.append([train_loss, train_acc])
                test_hist.append([test_loss, test_acc])
                
                # Save the network parameters if needed
                if (save_path is not None) and save_epoch:
                    torch.save(net.state_dict(), save_path+'params/params_after_epoch_' + str(epoch+1)+'.pth')
                    
            # Evaluation in the end of training (and saving results) is moved to other function
            # so if we set eval_epoch to None, the training is quite plain in the sense that 
            # we mostly don't do evaluation.
    
    # Return the dataset/batch statistics to plot learning curve
    if eval_epoch is not None:           
        return train_hist, test_hist, batch_hist
    else: 
        return None, None, batch_hist

# Plot Loss curve during training given a list of train/test loss values
def plot_learning_curve(train_hist, test_hist, batch_hist, plot_batch= False, 
                eval_epoch=10, num_epochs=100, desc = '', save_path = None):
    batch_hist = np.array(batch_hist)
    x = np.arange(0,num_epochs+1, eval_epoch) #later use as xticks
    
    if train_hist is not None:
        assert test_hist is not None
        train_hist = np.array(train_hist)
        test_hist = np.array(test_hist)
        fig1 = plt.figure(facecolor="w", figsize=(10, 5))
        plt.plot(x, train_hist[:,0])
        plt.plot(x, test_hist[:,0])
        plt.title('Loss curves ' + desc)
        plt.legend(["Train loss", "Test loss"])
        plt.xlabel("Epoch")
        plt.xticks(x)
        plt.ylabel("Loss")
        #fig1.savefig(save_path+'learning_curve.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        fig2 = plt.figure(facecolor="w", figsize=(10, 5))
        plt.plot(x, 100*train_hist[:,1])
        plt.plot(x, 100*test_hist[:,1])
        plt.title('Accuracy curves ' + desc)
        plt.legend(["Train accuracy", "Test accuracy"])
        plt.xlabel("Epoch")
        plt.xticks(x)
        plt.ylabel("%")
        #fig2.savefig(save_path+'learning_curve.png', bbox_inches='tight')
        plt.show()
        plt.close()
    
        if save_path is not None:
            fig1.savefig(save_path+'Loss_curve.png', bbox_inches='tight')
            fig2.savefig(save_path+'Accuracy_curve.png', bbox_inches='tight')
        # best_epoch = np.argmax()# Later choose the best epoch
        
    # later modify the plot for batches if needed
    if plot_batch:
        fig3 = plt.figure(facecolor="w", figsize=(10, 5))
        plt.plot(batch_hist[:,0])
        #plt.plot(batch_hist[:,1])
        plt.title('Minibatch loss ' + desc)
        #plt.legend(["Loss"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        
        if save_path is not None:
            fig3.savefig(save_path+'minibatch_curve.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
    
    
# compute and show the total loss/accuracy for the WHOLE data_loader in a given epoch
# train specifies whether the data_loader is train or test dataset. Just for printing
def evaluate_snn(net, data_loader, output='spike'):
    num_steps = net.num_steps
    if output=='mem':
        loss = nn.CrossEntropyLoss()
    elif output=='spike':
        loss = SF.ce_count_loss()
        #loss = SF.ce_rate_loss()
        
    # stores the total train/test loss in each epoch
    total_loss = 0.0
    # store the number of accurate predictions per minibatch
    acc, total = 0, 0
    
    with torch.no_grad():
        for data, targets in iter(data_loader):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
            
            net.eval()
            spk_rec, mem_rec = net(data)
            
            if output=='mem':
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], targets)
            else:
                loss_val = loss(spk_rec, targets)
            
            # compute the total train loss for plotting training curve
            batch_size = targets.size(0)
            total_loss += loss_val.item() * batch_size
            acc += SF.accuracy_rate(spk_rec, targets) * batch_size
            total += batch_size
            
    total_loss /= total
    accuracy = acc/total
    
    return total_loss, accuracy

def train_snn_monitor_grad(net, train_loader, test_loader, save_path=None, num_epochs=10,
                           output='spike', lr=5e-4, weight_decay = 0, monitor_grad=True):
    num_steps = net.num_steps # only used for membrane potential output
    if output=='mem':
        loss = nn.CrossEntropyLoss() # later take sum_t CE_loss(mem_out(t), label)
    elif output=='spike':
        loss = SF.ce_count_loss() #CE_loss(ave_t(spk_out(t)), label)
        #loss = SF.ce_rate_loss() #average over time of CE_loss(spk_out(t), label)
         
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # To display statistics
    #train_loss_hist, test_loss_hist = [], []

    # Outer training loop        
    for epoch in tqdm(range(num_epochs)):
        print(f"---------- Training epoch {epoch} ------------")
        # Minibatch training loop
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
    
            # forward pass
            net.train()
            spk_rec, mem_rec = net(data)
            
            if output=='mem':
            # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], targets)
            else: 
            #output=='spike'
                loss_val = loss(spk_rec, targets)
    
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        
            # Compute statistics within the minibatch
            if i%50==0:
                net.eval()
                if monitor_grad==True:
                    #total_params = 0
                    #total_grad_norm = 0.0
                    W0 = list(net.parameters())[0]
                    #print('Size of the first weight matrix', W0.grad.size(), 'Norm of the first weight matrix', W0.grad.norm().item())
                    # for param in net.parameters():
                    #     if param.grad is not None:
                    #         #print(f'Gradient norm {param.grad.norm().item()}, parameters size {param.numel()}')
                    #         total_grad_norm += param.grad.norm().item() ** 2  # Compute gradient norm
                    #         total_params += param.numel()
                    # total_grad_norm = total_grad_norm ** 0.5 
                acc = SF.accuracy_rate(spk_rec, targets)
                print(f"Iteration {i} --- Train Loss: {loss_val.item():.2f} --- Minibatch accuracy: {acc * 100:.2f}%")
                if monitor_grad==True:
                    print('Norm of the first weight matrix', W0.grad.norm().item())
                    
        # Save the parameters if needed
        if save_path is not None:
            if epoch == num_epochs-1: # if (epoch%5 == 0) or (epoch == num_epochs-1):            
                torch.save(net.state_dict(), save_path+'params_after_epoch_' + str(epoch+1)+'.pth')
            #train_loss_hist.append(train_loss)
            #test_loss_hist.append(test_loss)
        


def evaluate_ann(net, data_loader):
    loss = nn.CrossEntropyLoss()
        
    # stores the total train/test loss in each epoch
    total_loss = 0.0
    # store the number of accurate predictions per minibatch
    acc, total = 0, 0
    
    with torch.no_grad():
        for data, targets in iter(data_loader):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
            
            net.eval()
            out = net(data)
            loss_val = loss(out, targets)
            
            # compute the total train loss for plotting training curve
            batch_size = targets.size(0)
            total_loss += loss_val.item() * batch_size
            _, predicted = torch.max(out.data, 1)
            acc += (predicted == targets).sum().item() 
            total += batch_size
            
    total_loss /= total
    accuracy = acc/total
    
    return total_loss, accuracy
