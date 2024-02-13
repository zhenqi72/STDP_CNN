import torch     

class Later_inhibt(torch.nn.Module):
    
    def __init__(
        self,
        input_size,
    ):
        super(Later_inhibt, self).__init__()
        self.input_size = input_size
        self.mask = torch.ones(input_size)
        
        
    def forward(self, x, mask,m,v):
        
        if m == 0 :
            v = torch.tensor(v)
            #select the neurons that have the highest value in every 4 channels
            _,index = torch.max(input = v,dim=1)
            mask_temp = torch.zeros(v.shape)
            for i in range(index.shape[1]):
                for j in range(index.shape[2]):
                    mask_temp[:,index[0,i,j],i,j] = 1
            mask = mask * mask_temp
            mask = mask*x
            return x, mask,
        else:
            mask_temp = self.mask*x #create temparl mask to get fired neuron in this time 
            mask2 = mask_temp + mask # combine with previous mask to see which neuron were add or have to be inhibited
            result=torch.sum(mask2,1,keepdim=True) #according to the dimesnion 1 which is the channel size to sum all the feature mask
            position = torch.nonzero(result>1) #if the mask greater than 1, then this position neuron were inhibited
            dimension0 = position[:,-2]
            dimension1 = position[:,-1]
            dimension0=torch.tensor(dimension0)
            dimension1=torch.tensor(dimension1) #get which neuron should be inhibited
            mask_temp[:,:,dimension0,dimension1] = 0
            mask = mask+mask_temp
            
            return mask*x, mask
        