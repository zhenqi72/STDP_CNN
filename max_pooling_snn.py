import torch
class Max_pool_snn(torch.nn.Module):

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        mask
    ):
        super(Max_pool_snn, self).__init__()
        self.input_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mask = mask
        self.max_pool=torch.nn.MaxPool2d(self.input_size,self.pool_strd1)

    def forward(self,x):
        o_result = self.max_pool(x)
        mask = mask+x
        
        result = self.mask*o_result
        return result,mask