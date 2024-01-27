import torch.nn as nn
import torch
import numpy as np
class DoGFilter(nn.Module):
    def __init__(self, in_channels, kernel_size,sigma1,sigma2, stride=1, padding=(2,2)):
        super(DoGFilter, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        # initiate
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,padding=self.padding)
        #self.conv2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,padding=self.padding)
        self.weight1 = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size),requires_grad = False)
        #self.weight2 = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size),requires_grad = False)

        self.weight1.data  = self.DoG_kernel(self.sigma1, self.sigma2, self.kernel_size)#self.weight2.data
        self.conv1.weight = self.weight1
        #self.conv2.weight = self.weight2
        
        #create gaussin kernel 
    def DoG_kernel(self, s1, s2, size):
        r = np.arange(size)+1
        x = np.tile(r, [size, 1])
        y = x.T
        d2 = (x-size/2.-0.5)**2 + (y-size/2.-0.5)**2
        filt = 1/np.sqrt(2*np.pi) * (1/s1 * np.exp(-d2/(2*(s1**2))) - 1/s2 * np.exp(-d2/(2*(s2**2))))
        filt -= np.mean(filt[:])
        filt /= np.amax(filt[:])
        filt = torch.from_numpy(filt).view(1,1,5,5).float()
        print("filt is",filt)
        return filt          
        
    def forward(self, x):
        # create gaussin kernel
        x1 = self.conv1(x)
        #x2 = self.conv2(x)
        #x_on = x1 - x2 #on center filter 
        #x_off = x2 - x1#off center filter
        #x = torch.cat((x_on, x_off), dim=1)
        #print("x size",x.size())
        return x1