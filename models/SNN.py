import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn


#a single-layer SNN
class Net1(nn.Module):
    def __init__(self, num_steps=2):
        super().__init__()
        self.num_steps = num_steps
        # Initialize layers of widths 2-2-2
        self.fc1 = nn.Linear(2, 1, bias=True)
        self.fc1.weight.data.uniform_(-1, 3)
        self.lif1 = snn.Leaky(beta=1.0)
        
        #self.init_mem1 = 2*torch.rand(2)
        
    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        #mem1 = self.init_mem1

        # Record all layers
        spk1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
                        
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

        return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0)

# a simple SNN of architecture 2-2-2
class Net2(nn.Module):
    def __init__(self, num_steps=2):
        super().__init__()
        self.num_steps = num_steps
        # Initialize layers of widths 2-2-2
        self.fc1 = nn.Linear(2, 2, bias = True)
        self.fc1.weight.data.uniform_(0, 2)
        self.lif1 = snn.Leaky(beta=1.0)
        
        self.fc2 = nn.Linear(2, 1, bias = True)
        self.fc2.weight.data.uniform_(-1,3)
        self.lif2 = snn.Leaky(beta=1.0)
        
        #self.init_mem1 = 2*torch.rand(2)
        #self.init_mem2 = 2*torch.rand(2)
        
    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        #mem1 = self.init_mem1
        #mem2 = self.init_mem2
        
        # Record all layers
        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)