import torch
import torch.nn as nn

class AVNet(nn.Module):
    def __init__(self, small_kern=(3, 3), small_chan_1=16, small_chan_2=32, small_chan_3=32,
                 large_kern=(4, 4), large_chan_1=48, large_chan_2=96, large_chan_3=96,
                 fc_hidden_size=32, pool_kern=(2, 2), pool_stride=(1, 1)):
        super(AVNet, self).__init__()
        self.input_chan = 3
        self.small_kern = small_kern
        self.small_chan_1 = small_chan_1
        self.small_chan_2 = small_chan_2
        self.small_chan_3 = small_chan_3
        self.large_kern = large_kern
        self.large_chan_1 = large_chan_1
        self.large_chan_2 = large_chan_2
        self.large_chan_3 = large_chan_3
        self.fc_feat_num_1 = small_chan_2 + small_chan_3 + large_chan_2 + large_chan_3 + 4
        self.fc_feat_num_2 = fc_hidden_size
        self.pool_kern = pool_kern
        self.pool_stride = pool_stride

        self.pad = lambda tensor: nn.functional.pad(tensor, (1, 1, 1, 1), 'constant', 1)
        self.conv_small_1 = nn.Conv2d(self.input_chan, self.small_chan_1, self.small_kern)
        self.conv_small_2 = nn.Conv2d(self.small_chan_1, self.small_chan_2, self.small_kern)
        self.conv_small_3 = nn.Conv2d(self.small_chan_1, self.small_chan_3, self.small_kern)
        self.conv_large_1 = nn.Conv2d(self.input_chan, self.large_chan_1, self.large_kern)
        self.conv_large_2 = nn.Conv2d(self.large_chan_1, self.large_chan_2, self.large_kern)
        self.conv_large_3 = nn.Conv2d(self.large_chan_1, self.large_chan_3, self.large_kern)
        self.fc1 = nn.Linear(self.fc_feat_num_1, self.fc_feat_num_2)
        self.fc2 = nn.Linear(self.fc_feat_num_2, 1)

        self.nonlin = nn.ReLU()
        self.max_pool = nn.MaxPool2d(self.pool_kern, self.pool_stride)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, maps, action):
        maps = self.pad(maps)
        
        path_1 = self.nonlin(self.conv_small_1(maps))
        path_1_1 = self.nonlin(self.conv_small_2(path_1))
        path_1_2 = self.nonlin(self.conv_small_3(self.max_pool(path_1)))
        
        path_2 = self.nonlin(self.conv_large_1(maps))
        path_2_1 = self.nonlin(self.conv_large_2(path_2))
        path_2_2 = self.nonlin(self.conv_large_3(self.max_pool(path_2)))
        
        path_1_1 = self.global_max_pool(path_1_1)
        path_1_2 = self.global_max_pool(path_1_2)
        path_2_1 = self.global_max_pool(path_2_1)
        path_2_2 = self.global_max_pool(path_2_2)
        
        path_1_1 = torch.flatten(path_1_1, start_dim=1)
        path_1_2 = torch.flatten(path_1_2, start_dim=1)
        path_2_1 = torch.flatten(path_2_1, start_dim=1)
        path_2_2 = torch.flatten(path_2_2, start_dim=1)
        
        feats = torch.cat((path_1_1, path_1_2, path_2_1, path_2_2, action), dim=1)
        feats = self.nonlin(self.fc1(feats))
        return self.fc2(feats)
