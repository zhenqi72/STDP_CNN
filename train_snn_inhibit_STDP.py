from DoGFilter import DoGFilter
import torch
import numpy as np
import matplotlib.pyplot as plt

r"""
In this task, we train a spiking convolutional network to learn the
MNIST digit recognition task.
"""
from argparse import ArgumentParser
import os
import uuid

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
import torch.nn.functional as F


from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.iaf import IAFCell,IAFParameters,IAFFeedForwardState
from norse.torch.functional.stdp import STDPState,stdp_step_conv2d,STDPParameters
from norse.torch.functional.encode import spike_latency_encode

from DoGFilter_copy  import DoGFilter
from SVM_classifier import multiclass_hinge_loss,MultiClassSVM,binary_hinge_loss
from Lateral_inhibit import Later_inhibt

from STDP_selfmade import get_update_index,STDP_learning
from Max_pooling_snn import Max_pool_snn
import cv2

from sklearn import svm

          
class ConvNet_STDP(torch.nn.Module):
    """
    A convolutional network with LIF dynamics

    Arguments:
        num_channels (int): Number of input channels
        feature_size (int): Number of input features
        method (str): Threshold method
    """

    def __init__(
        self, 
        num_channels=1, 
        feature_size=[240,160], 
        method="super", 
        dtype=torch.float,
        batchsize=32,
        training = 0,
    ):
        super(ConvNet_STDP, self).__init__()
        #convolution parameters
        self.ken_size1 =5
        self.ken_size2 =16
        self.ken_size3 =5
        self.pool_strd1 =6
        self.pool_strd2 =2
        self.pool_size1 =7
        self.pool_size2 =2
        self.feature_size = 10 
        self.batchsize = batchsize
        
        self.fc1 = torch.nn.Linear(batchsize*10, 50)
        self.maxpool1 = Max_pool_snn(kernel_size=(self.pool_size1,self.pool_size1),stride=self.pool_strd1,padding=0)
        self.maxpool2 = Max_pool_snn(kernel_size=(self.pool_size2,self.pool_size2),stride=self.pool_strd2,padding=0)
        self.gpool = torch.nn.AdaptiveMaxPool2d((1))
        self.out = LILinearCell(50, 10)
        
        self.conv2d1 = nn.Conv2d(in_channels=num_channels,out_channels=4,kernel_size=self.ken_size1,bias = False)
        self.conv2d2 = nn.Conv2d(in_channels=4,out_channels=20,kernel_size=self.ken_size2,bias = False)
        self.conv2d3 = nn.Conv2d(in_channels=20,out_channels=self.feature_size,kernel_size=self.ken_size3,bias = False)
        
        #later layer
        self.later1 = Later_inhibt(input_size=[batchsize,4,236,156])
        self.later2 = Later_inhibt(input_size=[batchsize,20,24,10])
        self.later3 = Later_inhibt(input_size=[batchsize,10,8,1])
        
        self.w1 = torch.normal(mean=0.8, std=0.05, size=(4,1,5,5)) 
        self.w2 = torch.normal(mean=0.8, std=0.05, size=(20,4,16,16)) 
        self.w3 = torch.normal(mean=0.8, std=0.05, size=(10,20,5,5)) 
        self.conv2d1.weight = torch.nn.Parameter(self.w1, requires_grad=False)
        self.conv2d2.weight = torch.nn.Parameter(self.w2, requires_grad=False)
        self.conv2d3.weight = torch.nn.Parameter(self.w3, requires_grad=False)
        
        self.if1 = IAFCell(
            p=IAFParameters(method=method, alpha=torch.tensor(100.0),v_th= torch.tensor(10)),
        )
        self.if2 = IAFCell(
            p=IAFParameters(method=method, alpha=torch.tensor(100.0),v_th= torch.tensor(60)),
        )
        self.if3 = IAFCell(
            p=IAFParameters(method=method, alpha=torch.tensor(100.0),v_th= torch.tensor(np.inf)),
        )

        self.dtype = dtype
        self.training = training
        

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0, s1, s2, s3= None, None, None, None
        v1,v2,v3= None,None,None

        mask1 =torch.ones(self.batchsize,4,236,156)
        mask2 =torch.ones(self.batchsize,20,24,10) 
        mask3 =torch.ones(self.batchsize,10,8,1)  
        mask1_pool =torch.ones(self.batchsize,4, 39, 25)
        mask2_pool =torch.ones(self.batchsize,20, 12, 5) 
        for ts in range(seq_length):
            #dog filter
            with torch.no_grad():
                #first conv layer
                z2 = self.conv2d1(x[ts, :])
                #s is the voltage of neurons
                z2,s1,v1 = self.if1(z2, s1)             
                z2,mask1 = self.later1(z2,mask1,ts,v1)
                s1 = mask1*s1
                s1 = IAFFeedForwardState(s1)    
                z3,mask1_pool = self.maxpool1(z2,mask1_pool)
                #second conv layer
                z4 = self.conv2d2(z3)               
                z5, s2,v2 = self.if2(z4, s2)
                z5,mask2 = self.later2(z5,mask2,ts,v2) 
                s2 = mask2*s2
                s2 = IAFFeedForwardState(s2)     
                z6,mask2_pool = self.maxpool2(z5,mask2_pool)
                #third conv layer
                z7 = self.conv2d3(z6)                
                z8, s3,v3 = self.if3(z7,s3)                
                _,mask3 = self.later3(z8,mask3,ts,v3)
                s3 = mask3*s3
                s3 = IAFFeedForwardState(s3)
                z9 = self.gpool(v3)     
                if self.training == 1:
                    maxvel1,maxind11,maxind21= get_update_index(v1,mask1)  
                    self.w1  = STDP_learning(S_pre_sz=x[ts,:].shape,s_pre=x[ts,:], s_cur=z2, w=self.w1, threshold=10,  # Input arrays
                    maxval=maxvel1, maxind1=maxind11, maxind2=maxind21,  # Indices
                    stride=1, a_plus=0.004,a_minus=0.003) 
                    self.conv2d1.weight = torch.nn.Parameter(self.w1, requires_grad=False)
                    
                    maxvel2,maxind12,maxind22= get_update_index(v2,mask2)  
                    self.w2  = STDP_learning(S_pre_sz=z3.shape,s_pre=z3, s_cur=z5, w=self.w2, threshold=60,  # Input arrays
                    maxval=maxvel2, maxind1=maxind12, maxind2=maxind22,  # Indices
                    stride=1, a_plus=0.004,a_minus=0.003)
                    self.conv2d2.weight = torch.nn.Parameter(self.w2, requires_grad=False)

                    maxvel3,maxind13,maxind23= get_update_index(v3,mask3) 
                    self.w3  = STDP_learning(S_pre_sz=z6.shape,s_pre=z6, s_cur=z8, w=self.w3, threshold=2,
                    maxval=maxvel3, maxind1=maxind13, maxind2=maxind23, 
                    stride=1, a_plus=0.004,a_minus=0.003)
                    self.conv2d3.weight = torch.nn.Parameter(self.w3, requires_grad=False)
                        
        return z9,self.w1,self.w2,self.w3

class IF_Model(torch.nn.Module):
    def __init__(
        self,
        input_features,
        seq_length,
        input_scale,
        model="super",
        only_first_spike=False,
        batchsize = 32,
    ):
        super(IF_Model, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        self.cnn = ConvNet_STDP(method=model,batchsize=batchsize,training=1)
        self.seq_length = seq_length
        self.input_scale = input_scale

    def forward(self, x):
        batch_size = x.shape[0]
        #print("x.shape",x.shape)
        x = self.constant_current_encoder(
            x.view(-1,self.input_features) * self.input_scale
        )
        x = spike_latency_encode(x)
        x = x.reshape(self.seq_length, batch_size, 1, 240, 160)
        features,w1,w2,w3 = self.cnn(x)
        return features,w1,w2,w3

def train_snn(
    model,
    device,
    train_loader,
    batchsize,
):
    #model.train()
    dogfilter = DoGFilter(in_channels=1, sigma1=1,sigma2=2,kernel_size=5)
    ac = []
    x = []
    y = []
    for batch_idx, (data, target) in enumerate(train_loader):  
        if len(data) !=  batchsize:
            continue    
        data = dogfilter(data)
        if target == 3:
            target = torch.tensor(1)    
        data, target = data.to(device), target.to(device)
        output,w1,w2,w3 = model(data)
        #output = output.view(1,2)
        x.append(output)
        y.append(target)
        print("epoch",batch_idx)
        """
        optimizer.zero_grad()
        loss = binary_hinge_loss(output, target)
        loss.backward()
        optimizer.step()
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        ac.append(accuracy)
        ac = []
        """
    x_train = np.array(x[0:1000])
    y_train = np.array(y[0:1000]) 
    x_test = np.array(x[1000:])
    y_test = np.array(y[1000:]) 
    clf = svm.SVC(C=1.0,  gamma='auto')
    clf.fit(x_train,y_train)
    # Obtain the Training Error
    score=clf.score(x_test, y_test)
    print("accuracy is {}".format(score))         
        
    return score,w1,w2

def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "is_best": is_best,
        },
        path,
    )


def load(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    return model, optimizer


def main(args):
    writer = SummaryWriter()
    torch.set_printoptions(threshold=100000)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True} if args.device == "cuda" else {}
    
    dataset = torchvision.datasets.Caltech101(
            root=".",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.PILToTensor(),
                    torchvision.transforms.Resize((240,160)),
                    torchvision.transforms.Grayscale(1)                    
                ]
            ),
        )
    
    selected_classes = [0,3]#67
    index = []

    for i,(_,label) in enumerate(dataset):
        if label in selected_classes:
            index.append(i)
    sub_dataset = torch.utils.data.Subset(dataset, index)

    train_loader = torch.utils.data.DataLoader(
        sub_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )

    input_features = 240 * 160

    model = IF_Model(
        input_features,
        args.seq_length,
        input_scale=args.input_scale,
        model=args.method,
        only_first_spike=args.only_first_spike,
        batchsize=args.batch_size,
    ).to(device)
    

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.only_output:
        optimizer = torch.optim.Adam(model.out.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        accuracy,w1,w2 = train_snn(
            model,
            device,
            train_loader,
            batchsize=args.batch_size,
        )
    np.save("accuracy,npy",np.array(accuracy))
    np.save("w1.npy", np.array(w1))
    np.save("w2.npy", np.array(w2))
       

    model_path = "caltech-snn.pt"    
    save(
        model_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        "MNIST digit recognition with convolutional SNN. Requires Tensorboard, Matplotlib, and Torchvision"
    )
    parser.add_argument(
        "--only-first-spike",
        type=bool,
        default=False,
        help="Only one spike per input (latency coding).",
    )
    parser.add_argument(
        "--save-grads",
        type=bool,
        default=False,
        help="Save gradients of backward pass.",
    )
    parser.add_argument(
        "--grad-save-interval",
        type=int,
        default=10,
        help="Interval for gradient saving of backward pass.",
    )
    parser.add_argument(
        "--refrac", type=bool, default=False, help="Use refractory time."
    )
    parser.add_argument(
        "--plot-interval", type=int, default=10, help="Interval for plotting."
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Scaling factor for input current.",
    )
    parser.add_argument(
        "--find-learning-rate",
        type=bool,
        default=False,
        help="Use learning rate finder to find learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use by pytorch.",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training episodes to do."
    )
    parser.add_argument(
        "--seq-length", type=int, default=30, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of examples in one minibatch.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="super",
        choices=["super", "tanh", "circ", "logistic", "circ_dist"],
        help="Method to use for training.",
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix to use for saving the results"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--clip-grad",
        type=bool,
        default=False,
        help="Clip gradient during backpropagation",
    )
    parser.add_argument(
        "--grad-clip-value", type=float, default=1.0, help="Gradient to clip at."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3, help="Learning rate to use."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="In which intervals to display learning progress.",
    )
    parser.add_argument(
        "--model-save-interval",
        type=int,
        default=15,
        help="Save model every so many epochs.",
    )
    parser.add_argument(
        "--save-model", type=bool, default=True, help="Save the model after training."
    )
    parser.add_argument("--big-net", type=bool, default=False, help="Use bigger net...")
    parser.add_argument(
        "--only-output", type=bool, default=False, help="Train only the last layer..."
    )
    parser.add_argument(
        "--do-plot", type=bool, default=False, help="Do intermediate plots"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1234, help="Random seed to use"
    )
    args = parser.parse_args()
    main(args)
