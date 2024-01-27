import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision

from norse.torch.models.conv import ConvNet4
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.iaf import IAFCell,IAFParameters
import torch.nn.functional as F
from norse.torch.functional.stdp import (STDPState,stdp_step_conv2d,STDPParameters)
import cv2

class Conv2d_Spike(nn.Module):
    
    def __init__(self,):
        super(Conv2d_Spike, self).__init__()
        self.iafparams = IAFParameters()
        self.iafcell = IAFCell()

    def forward(self,x):
