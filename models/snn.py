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

# C_first_hidden = number of channels in first hidden layer
# C_hidden = number of 
class ConvSNN(nn.Module):
    def __init__(self, num_steps, n_in, n_out, C_first_hidden, ):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3