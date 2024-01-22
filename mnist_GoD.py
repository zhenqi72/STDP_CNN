r"""
In this task, we train a spiking convolutional network to learn the
MNIST digit recognition task.
"""
from argparse import ArgumentParser
import os
import uuid

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision

from norse.torch.models.conv import ConvNet4
from norse.torch.module.encode import ConstantCurrentLIFEncoder

import torch.nn as nn
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
import torch.nn.functional as F
from norse.torch.functional.stdp import (STDPState,stdp_step_conv2d,STDPParameters)


class DoGFilter(nn.Module):
    def __init__(self, in_channels, kernel_size,sigma1,sigma2, stride=1, padding=0):
        super(DoGFilter, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        # initiate
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5)
        self.weight1 = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size),requires_grad = False)
        self.weight2 = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size),requires_grad = False)
        
        #create gaussin kernel 
    def DoG_kernel(self, sigma1, sigma2, size):
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax)
        g1 = torch.exp(-(xx**2 + yy**2) / (2 * sigma1**2))
        g1 = g1/g1.sum()
        g2 = torch.exp(-(xx**2 + yy**2) / (2 * sigma2**2))
        g2 = g2/g2.sum()
        # transfer to Tensor
        g1 = g1.view(1,1,5,5)
        g2 = g2.view(1,1,5,5)
        
        #print("g1",g1)
        #print("g2",g2)
        return g1,g2          
        
    def forward(self, x):
        # create gaussin kernel
        self.weight1.data, self.weight2.data = self.DoG_kernel(self.sigma1, self.sigma2, self.kernel_size)
        self.conv1.weight = self.weight1
        self.conv2.weight = self.weight2
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_on = x1 - x2 #on center filter 
        x_off = x2 - x1#off center filter
        x = torch.cat((x_on, x_off), dim=1)
        #print("x size",x.size())
        return x
          
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
        feature_size=24, 
        method="super", 
        dtype=torch.float
    ):
        super(ConvNet_STDP, self).__init__()
        self.dogfilter = DoGFilter(in_channels=num_channels, sigma1=1,sigma2=2,kernel_size=5)
        self.features = int(((feature_size - 4) / 2 - 4) / 2)
        self.fc1 = torch.nn.Linear(3200, 500)
        self.gpool = torch.nn.AdaptiveMaxPool2d((1))
        self.out = LILinearCell(500, 10)
        
        self.conv2d1 = nn.Conv2d(in_channels=2,out_channels=30,kernel_size=5,bias = False)
        self.conv2d2 = nn.Conv2d(in_channels=30,out_channels=100,kernel_size=5,bias = False)
        
        self.stdp_param1 = STDPParameters(
            eta_plus=0.004,
            eta_minus=0.003,
            stdp_algorithm="multiplicative_relu",
            convolutional = True,
            mu=0)
        self.stdp_param2 = STDPParameters(
            eta_plus=0.004,
            eta_minus=0.003,
            stdp_algorithm="multiplicative_relu",
            convolutional = True,
            mu=0)
        self.w1 = torch.randn(30,2,5,5)        
        #self.w1 = torch.unsqueeze(self.w1,0)
        self.w2 = torch.randn(100,30,5,5)
        #self.w2 = self.w2.view(100,30,5,5)
        self.conv2d1.weight = torch.nn.Parameter(self.w1, requires_grad=False)
        self.conv2d2.weight = torch.nn.Parameter(self.w2, requires_grad=False)
        
        self.lif0 = LIFCell(
            p=LIFParameters(method=method, alpha=torch.tensor(100.0),v_th= 15),
        )
        self.lif1 = LIFCell(
            p=LIFParameters(method=method, alpha=torch.tensor(100.0),v_th= 10),
        )
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=torch.tensor(100.0)))

        self.dtype = dtype
        

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0, s1, s2, s3,s4, so = None, None, None, None,None,None
        t_pre1,t_pre2,t_post1,t_post2 = None,None,None,None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=self.dtype
        )
        # ConvNet_STDP
        with torch.no_grad():
            for ts in range(seq_length):
                #dog filter
                z = self.dogfilter(x[ts, :])                
                #first conv layer
                z2 = self.conv2d1(z)
                z2, s1 = self.lif0(z2, s1)
                if t_pre1 == None:
                    t_pre1 =torch.zeros((z.shape[0], z.shape[1], z.shape[2], z.shape[3]))
                if t_post1 == None:
                    t_post1 =  torch.zeros((z2.shape[0], z2.shape[1], z2.shape[2], z2.shape[3]))
                                                       
                self.w1,_  = stdp_step_conv2d(z,z2,self.w1,STDPState(t_pre1,t_post1),p_stdp= self.stdp_param1)
                self.conv2d1.weight = torch.nn.Parameter(self.w1, requires_grad=False) 

                #max pooling layer          
                z3 = torch.nn.functional.max_pool2d(z2, kernel_size = 2, stride = 2)

                #second conv layer
                z4 = 10 * self.conv2d2(z3)
                z4, s2 = self.lif1(z4, s2)
                if t_pre2 == None:
                    t_pre2 =torch.zeros((z3.shape[0], z3.shape[1], z3.shape[2], z3.shape[3]))
                if t_post2== None:
                    t_post2 =  torch.zeros((z4.shape[0], z4.shape[1], z4.shape[2], z4.shape[3]))
                                    
                self.w2,_ = stdp_step_conv2d(z3,z4,self.w2,STDPState(t_pre2,t_post2),p_stdp= self.stdp_param2)
                self.conv2d2.weight = torch.nn.Parameter(self.w2, requires_grad=False)

                #global pooling layer
                z4 = self.gpool(z4)
                z4 = z4.view(-1,3200)
                #print("size of z4",z4.size())
                #full connect layer
                z4 = self.fc1(z4)
                z4, s3 = self.lif2(z4, s3)
                v, so = self.out(torch.nn.functional.relu(z4), so)
                voltages[ts, :, :] = v
        return voltages

class LIFConvNet(torch.nn.Module):
    def __init__(
        self,
        input_features,
        seq_length,
        input_scale,
        model="super",
        only_first_spike=False,
    ):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        self.rsnn = ConvNet_STDP(method=model)
        self.seq_length = seq_length
        self.input_scale = input_scale

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.constant_current_encoder(
            x.view(-1, self.input_features) * self.input_scale
        )
        if self.only_first_spike:
            # delete all spikes except for first
            zeros = torch.zeros_like(x.cpu()).detach().numpy()
            idxs = x.cpu().nonzero().detach().numpy()
            spike_counter = np.zeros((batch_size, 28 * 28))
            for t, batch, nrn in idxs:
                if spike_counter[batch, nrn] == 0:
                    zeros[t, batch, nrn] = 1
                    spike_counter[batch, nrn] += 1
            x = torch.from_numpy(zeros).to(x.device)

        x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
        voltages = self.rsnn(x)
        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y

def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    clip_grad,
    grad_clip_value,
    epochs,
    log_interval,
    do_plot,
    plot_interval,
    seq_length,
    writer,
):
    model.train()
    losses = []

    batch_len = len(train_loader)
    step = batch_len * epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        step += 1

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    epochs,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        if step % log_interval == 0:
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", accuracy.item(), step)

            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram(tag, value.data.cpu().numpy(), step)
                if value.grad is not None:
                    writer.add_histogram(
                        tag + "/grad", value.grad.data.cpu().numpy(), step
                    )

        if do_plot and batch_idx % plot_interval == 0:
            ts = np.arange(0, seq_length)
            fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
            axs = axs.reshape(-1)  # flatten
            for nrn in range(10):
                one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
                fig.sca(axs[nrn])
                fig.plot(ts, one_trace)
            fig.xlabel("Time [s]")
            fig.ylabel("Membrane Potential")

            writer.add_figure("Voltages/output", fig, step)

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, test_loader, epoch, method, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set {method}: Average loss: {test_loss:.4f}, \
            Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
    )
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)

    return test_loss, accuracy


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

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True} if args.device == "cuda" else {}
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        **kwargs,
    )

    input_features = 28 * 28

    model = LIFConvNet(
        input_features,
        args.seq_length,
        input_scale=args.input_scale,
        model=args.method,
        only_first_spike=args.only_first_spike,
    ).to(device)
    

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.only_output:
        optimizer = torch.optim.Adam(model.out.parameters(), lr=args.learning_rate)

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(args.epochs):
        training_loss, mean_loss = train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            clip_grad=args.clip_grad,
            grad_clip_value=args.grad_clip_value,
            epochs=args.epochs,
            log_interval=args.log_interval,
            do_plot=args.do_plot,
            plot_interval=args.plot_interval,
            seq_length=args.seq_length,
            writer=writer,
        )
        test_loss, accuracy = test(
            model, device, test_loader, epoch, method=args.method, writer=writer
        )

        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        max_accuracy = np.max(np.array(accuracies))

        if (epoch % args.model_save_interval == 0) and args.save_model:
            model_path = f"mnist-{epoch}.pt"
            save(
                model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                is_best=accuracy > max_accuracy,
            )

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("test_losses.npy", np.array(test_losses))
    np.save("accuracies.npy", np.array(accuracies))
    model_path = "mnist-final.pt"
    save(
        model_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        is_best=accuracy > max_accuracy,
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
        "--epochs", type=int, default=10, help="Number of training episodes to do."
    )
    parser.add_argument(
        "--seq-length", type=int, default=200, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        default=50,
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
