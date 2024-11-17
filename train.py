import torch
import snntorch.functional as SF
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
path = './mnist_results/'


# Let's set output to be 'spike', we may think about mem_output later. 
def train_snn(net, train_loader, test_loader, save_path=None, num_epochs=10, output='spike'):
    num_steps = net.num_steps # only used for membrane potential output
    if output=='mem':
        loss = nn.CrossEntropyLoss() # later take sum_t CE_loss(mem_out(t), label)
    elif output=='spike':
        loss = SF.ce_count_loss() #CE_loss(ave_t(spk_out(t)), label)
        #loss = SF.ce_rate_loss() #average over time of CE_loss(spk_out(t), label)
         
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

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
                acc = SF.accuracy_rate(spk_rec, targets)
                print(f"Iteration {i} --- Train Loss: {loss_val.item():.2f} --- Minibatch accuracy: {acc * 100:.2f}%\n")
                
        # Compute statistics over the whole train/test dataset. For now we compute only the last epoch, 
        # i.e. after training. Later we might change this to plot the learning curve.
        
        if epoch == num_epochs-1: # if (epoch%5 == 0) or (epoch == num_epochs-1):
            print(f"####### Statistics over the whole train/test dataset after epoch {epoch} #######")
            file = None
            if save_path is not None:
                file = open(save_path+'results.txt','a')
                torch.save(net.state_dict(), save_path+'params_at_epoch_' + str(epoch)+'.pth')
            print_snn_statistics(net, train_loader, epoch=epoch, file=file, train=True, output=output)
            print_snn_statistics(net, test_loader, epoch=epoch, file=file, train=False, output=output)
            if file is not None:
                file.close()
            #train_loss_hist.append(train_loss)
            #test_loss_hist.append(test_loss)
        
    

# Plot Loss
def plot_learning_curve(train_loss_hist, test_loss_hist, path=path):
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(train_loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss curves")
    plt.legend(["Train Loss", "Test Loss"])
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig.savefig(path+'learning_curve.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    
# compute and show the total loss/accuracy for the WHOLE data_loader in a given epoch
# train specifies whether the data_loader is train or test dataset. Just for printing
def print_snn_statistics(net, data_loader, epoch, file=None, train=True, output='spike'):
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
    if train:
        print_and_save(f"Train loss: {total_loss:.2f}, train accuracy: {acc/total *100:.2f} % ", file)    
    else:
        print_and_save(f"Test loss: {total_loss:.2f}, test accuracy: {acc/total *100:.2f} %", file)





def print_and_save(text_str, file):
    print(text_str)
    if file is not None:
        print(text_str, file=file)
