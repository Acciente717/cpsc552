import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality

class FCNN(nn.Module):
    def __init__(self, input_dim=154, output_dim=1, n_hidden_1=100, n_hidden_2=100, n_hidden_3=100):
        super(FCNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, output_dim)

        self.nonlin1 = nn.ReLU()
        self.nonlin2 = nn.ReLU()
        self.nonlin3 = nn.ReLU()

    def forward(self, maps, action):
        maps = maps.flatten(start_dim=1)
        combined =  torch.cat([maps, action], dim=1)
        h1 = self.nonlin1(self.layer1(combined))
        h2 = self.nonlin2(self.layer2(h1))
        h3 = self.nonlin3(self.layer3(h2))
        output = self.layer4(h2)
        return output
