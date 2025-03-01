import snntorch as snn
import torch
import torch.nn as nn
from snntorch import utils


# data always appear in form [time x batch x spatial_dimension]
# choose n_in/n_out based on datasets
# n_first_hidden = number of neurons in first hidden layer
# n_hidden = number of neurons in each later spike layer
# num_binary_layers = number of spike layers
# output = [spike_train, mem_train]
    
class SNN(nn.Module):
    def __init__(self, num_steps=2, n_in=28*28, n_out=10, n_first_hidden=20, 
                 num_binary_layers = 2, n_hidden = 10):
        super().__init__()
        self.num_steps = num_steps
        
        layers = [nn.Linear(n_in, n_first_hidden), 
                  snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True)]
        
        if num_binary_layers <2: 
            raise Exception("Number of binary layers must be greater equal 2")
        elif num_binary_layers == 2:
            layers+=[nn.Linear(n_first_hidden, n_out), 
                     snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True,output=True)]
        else:
            layers+=[nn.Linear(n_first_hidden, n_hidden), 
                     snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True)]
            for i in range(num_binary_layers-3):
                layers+= [nn.Linear(n_hidden, n_hidden), 
                          snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True)]
            layers+=[nn.Linear(n_hidden, n_out), 
                     snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True, output=True)]
        
        self.net = nn.Sequential(*layers)
        
        
    def forward(self, x):
        spk_rec = [] # record spikes over time
        mem_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        for step in range(self.num_steps):
            spk_out, mem_out = self.net(x.flatten(1))
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)

# We consider the architecture 
# (C_in x H x W) -> (C_1 x H x W) -> (C_2 x H x W)... -> (C_2 x H x W) -> linear -> out
# C_first_hidden = number of channels in first hidden layer
# C_hidden = number of channels in subsequent layer
class CSNN(nn.Module):
    def __init__(self, num_steps, chw_in, n_out, C_first_hidden, num_binary_layers, C_hidden):
        super().__init__()  
        self.num_steps = num_steps
        wh = chw_in[1]*chw_in[2]
        layers = [nn.Conv2d(in_channels=chw_in[0], out_channels=C_first_hidden, kernel_size=3, stride=1, padding=1), 
                  snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True)]
        
        if num_binary_layers <2: 
            raise Exception("Number of binary layers must be greater equal 2")
        elif num_binary_layers == 2:
            layers+=[nn.Flatten(), nn.Linear(C_first_hidden * wh, n_out), 
                     snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True,output=True)]
        else:
            layers+=[nn.Conv2d(in_channels= C_first_hidden, out_channels=C_hidden, kernel_size=3, stride=1, padding=1), 
                      snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True)]
            for i in range(num_binary_layers-3):
                layers+=[nn.Conv2d(in_channels= C_hidden, out_channels=C_hidden, kernel_size=3, stride=1, padding=1), 
                          snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True)]
            layers+=[nn.Flatten(), nn.Linear(C_hidden * wh, n_out), 
                     snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True,init_hidden=True, output=True)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        spk_rec = [] # record spikes over time
        mem_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        for step in range(self.num_steps):
            spk_out, mem_out = self.net(x)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)