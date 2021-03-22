import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality

import torchvision

import numpy as np

class Q_Network(nn.Module):

    def __init__(self, BatchSize, MapHeight, MapWidth, Covn1OutChan, Conv1Kernel, Covn2OutChan, Conv2Kernel, HiddenSize):
        super(Q_Network, self).__init__()
        
        # input parameters    x[batch size, channel number, height, weight]
        self.input_BatchSize  = BatchSize
        self.input_ChannelNum = 3  # three 2D maps: obstacle, tgt, agent 
        self.input_MapHeight  = MapHeight
        self.input_MapWidth   = MapWidth
    
        # Conv 2D layer 1    
        self.conv1_InChannelNum = self.input_ChannelNum
        self.conv1_OutChannelNum = Covn1OutChan
        self.conv1_KernelSize = Conv1Kernel
        
        # batch normal layer 1
        self.bn1_FeaturesNum = self.conv1_OutChannelNum
        
        # activation 1
        
        # Conv 2D layer 2
        self.conv2_InChannelNum = self.conv1_OutChannelNum
        self.conv2_OutChannelNum = Covn2OutChan
        self.conv2_KernelSize = Conv2Kernel
        
        # batch normal layer 2
        self.bn2_FeaturesNum = self.conv2_OutChannelNum
        
        # activation 2
        
        # flatten layer
                
        # Number of Linear input connections depends on output of conv2d layers
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - kernel_size + 1) 
        conv_width = conv2d_size_out(conv2d_size_out(self.input_MapWidth))
        conv_height = conv2d_size_out(conv2d_size_out(self.input_MapHeight))
        
        # dense layer
        self.linear_input_dim = conv_width * conv_height * self.bn2_FeaturesNum * self.input_BatchSize

        self.n_hidden_1 = HiddenSize
        self.n_hidden_2 = HiddenSize
        self.n_hidden_out = 1  # value network
        
        self.conv1 = nn.Conv2d(self.conv1_InChannelNum, self.conv1_OutChannelNum, self.conv1_KernelSize)
        self.bn1 = nn.BatchNorm2d(self.bn1_FeaturesNum)
        self.nonlin1 = nn.ReLU(inplace = True)
        
        self.conv2 = nn.Conv2d(self.conv2_InChannelNum, self.conv2_OutChannelNum, self.conv2_KernelSize)
        self.bn2 = nn.BatchNorm2d(self.bn2_FeaturesNum)
        self.nonlin2 = nn.ReLU(inplace = True)
        
        #self.flatten = torch.flatten()
        
        self.linear_layer1 = nn.Linear(self.linear_input_dim + 4, self.n_hidden_1)
        self.nonlin3 = nn.ReLU()

        self.linear_layer2 = nn.Linear(self.n_hidden_1, self.n_hidden_2)
        self.nonlin4 = nn.ReLU()
        
        self.linear_layer3 = nn.Linear(self.n_hidden_2, self.n_hidden_out)
        self.nonlin5 = nn.ReLU()
        

    def forward(self, x1, x2): # x1: maps, x2: action
        
        h1 = self.nonlin1(self.bn1(self.conv1(x1)))
        h2 = self.nonlin2(self.bn2(self.conv2(h1)))

        h3 = torch.flatten(h2)
        h3 = torch.cat((h3, x2))

        h4 = self.nonlin3(self.linear_layer1(h3))
        h5 = self.nonlin4(self.linear_layer2(h4))
        out = self.nonlin5(self.linear_layer3(h5))

        return out
        

'''
# maps is three channel 2D maps
# [BatchSize, input_ChannelNum = 3, MapHeight, MapWidth] 
input1 = torch.randn(1, 3, 10, 10) 
# action is a vector  
# [1, 0, 0, 0] means go up, [0, 1, 0, 0] means go down, [0, 0, 1, 0] means go left, [0, 0, 0, 1] means go right
input2 = torch.randn(4)

model = Q_Network(BatchSize = 1, MapHeight = 10, MapWidth = 10, Covn1OutChan = 3, Conv1Kernel = 3, Covn2OutChan = 2, Conv2Kernel = 3, HiddenSize = 50)
reward = model(x1 = input1, x2 = input2)
print(reward)
'''