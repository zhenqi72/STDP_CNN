import torch
class Max_pool_snn(torch.nn.Module):

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
    ):
        super(Max_pool_snn, self).__init__()
        self.input_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_pool=torch.nn.MaxPool2d(self.input_size,self.stride)

    def forward(self,x,mask):#mask size should equal to o_result not x
        o_result = self.max_pool(x)
        result = mask*o_result
        #at first the mask is all ones means all the neurons can fire
        fired = torch.where(o_result == 1)
        mask[fired] = 0        
        
        return result,mask#