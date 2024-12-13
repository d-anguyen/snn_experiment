#import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, n_in, n_out, n_first_hidden, num_hidden_layers, n_hidden):
        super().__init__()
        
        layers = [nn.Linear(n_in, n_first_hidden), nn.ReLU()]
        
        if num_hidden_layers <1: 
            raise Exception("Number of hidden layers must be greater equal 1")
        elif num_hidden_layers == 1:
            layers+=[nn.Linear(n_first_hidden, n_out)]
        else:
            layers+=[nn.Linear(n_first_hidden, n_hidden), nn.ReLU()]
            for i in range(num_hidden_layers-2):
                layers+= [nn.Linear(n_hidden, n_hidden),nn.ReLU()]
            layers+=[nn.Linear(n_hidden, n_out)]
        
        self.net = nn.Sequential(*layers)
        
        
    def forward(self, x):
        out = self.net(x.flatten(1))
        return out

class CNN(nn.Module):
    def __init__(self, chw_in, n_out, C_first_hidden, num_hidden_layers, C_hidden, pool = False):
        super().__init__()  
        wh = chw_in[1]*chw_in[2]
        print(wh)
        
        layers = [nn.Conv2d(in_channels=chw_in[0], out_channels=C_first_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        if pool:
            layers+=[nn.MaxPool2d(kernel_size=2, stride=2)]
            
        if num_hidden_layers <1: 
            raise Exception("Number of hidden layers must be greater equal 1")
        elif num_hidden_layers == 1:
            layers+=[nn.Flatten(), nn.Linear(C_first_hidden * wh, n_out), nn.ReLU()]
        else:
            layers+=[nn.Conv2d(in_channels= C_first_hidden, out_channels=C_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU()]
            
            for i in range(num_hidden_layers-2):
                layers+=[nn.Conv2d(in_channels= C_hidden, out_channels=C_hidden, kernel_size=3, stride=1, padding=1), nn.ReLU()]
            
            if pool: 
                layers+=[nn.MaxPool2d(kernel_size=2, stride=2)]
                layers+=[nn.Flatten(), nn.Linear(C_hidden * int(wh/16), n_out)]
            else: 
                layers+=[nn.Flatten(), nn.Linear(C_hidden * wh, n_out)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.net(x)
        return out

