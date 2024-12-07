#import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, n_in=28*28, n_out=10, n_first_hidden=20, 
                 num_hidden_layers = 2, n_hidden = 10):
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

class Sigmoid_ANN(nn.Module):
    def __init__(self, n_in=28*28, n_out=10, n_first_hidden=20, 
                 num_hidden_layers = 2, n_hidden = 10):
        super().__init__()
        
        layers = [nn.Linear(n_in, n_first_hidden), nn.Sigmoid()]
        
        if num_hidden_layers <1: 
            raise Exception("Number of hidden layers must be greater equal 1")
        elif num_hidden_layers == 1:
            layers+=[nn.Linear(n_first_hidden, n_out)]
        else:
            layers+=[nn.Linear(n_first_hidden, n_hidden), nn.Sigmoid()]
            for i in range(num_hidden_layers-2):
                layers+= [nn.Linear(n_hidden, n_hidden),nn.Sigmoid()]
            layers+=[nn.Linear(n_hidden, n_out)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.net(x.flatten(1))
        return out
