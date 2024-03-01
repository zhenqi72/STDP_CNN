import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

def DoG_encoder(img,num_layers,total_time):
    img = img.view(240,160)
    img = img.numpy()
    I = np.argsort(1 / img.flatten())  # Get indices of sorted latencies
    lat = np.sort(1 / img.flatten())  # Get sorted latencies
    I = np.delete(I, np.where(lat == np.inf))  # Remove infinite latencies indexes
    II = np.unravel_index(I, img.shape)  # Get the row, column and depth of the latencies in order
    t_step = np.ceil(np.arange(I.size) / ((I.size) / (total_time - num_layers))).astype(np.uint8)
    II += (t_step,)
    spike_times = np.zeros((img.shape[0], img.shape[1], total_time))
    spike_times[II] = 1
    spike_times = torch.tensor(spike_times)
    spike_times = spike_times.view(1,1,spike_times.shape[0],spike_times.shape[1],spike_times.shape[2])
    return spike_times

class DoGFilter(nn.Module):
    def __init__(self, in_channels, kernel_size,sigma1,sigma2):
        super(DoGFilter, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
                
        
    def forward(self, x):
        # create gaussin kernel2
        x1 = gaussian_filter(x,self.sigma1,radius=3)
        x2 = gaussian_filter(x,self.sigma2,radius=3)
    
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)
        
        x_on = x1 - x2 #on center filter 
        x_off = x2 - x1#off center filter
        x = torch.cat((x_on, x_off), dim=1)
        x_on = DoG_encoder(x_on,total_time=15,num_layers=6)
        #print("x size",x.size())
        return x_on