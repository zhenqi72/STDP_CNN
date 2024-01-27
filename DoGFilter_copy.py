import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

class DoGFilter(nn.Module):
    def __init__(self, in_channels, kernel_size,sigma1,sigma2):
        super(DoGFilter, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
                
        
    def forward(self, x):
        # create gaussin kernel2
        x1 = gaussian_filter(x,self.sigma1)
        x2 = gaussian_filter(x,self.sigma2)
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)
        x_on = x1 - x2 #on center filter 
        x_off = x2 - x1#off center filter
        x = torch.cat((x_on, x_off), dim=1)
        #print("x size",x.size())
        return x