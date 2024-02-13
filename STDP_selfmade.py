import numpy as np
import torch

def get_update_index(v,mask):
    v = v*mask #get the neuron that can fire for each layer
    maxvel,index = torch.max(input=v.view(v.shape[0],v.shape[1],-1),dim=2)
    mask = mask.view(mask.shape[0],mask.shape[1],-1)
    print("mask in {} is{}".format(index[0][0:2],mask[:,:,index[0][0:2]]))
    #print("index is",index)
    maxind1 = torch.squeeze(index // v.shape[3],0)
    maxind2 = torch.squeeze(index %  v.shape[3],0)
    return maxvel,maxind1,maxind2

def delta_t(fired,ss):
    d_t = fired*ss
    return d_t

def STDP_learning(S_pre_sz,s_pre, s_cur, w, threshold,  # Input arrays
                  maxval, maxind1, maxind2,  # Indices
                  stride, a_minus, a_plus):  # Parameter
    for i in range(w.shape[0]):
        if maxval[i]>threshold:
            # Select the input  
            input = torch.zeros(w[i,:, :, :].shape)
            #ss is 3 dimensions
            if maxind2[i]*stride >= S_pre_sz[3] - w.shape[3] and maxind1[i]*stride >= S_pre_sz[2] - w.shape[2]:
                ss = s_pre[0,:,maxind1[i] * stride:, maxind2[i] * stride:]
                input[:,:ss.shape[1], :ss.shape[2] ] = torch.tensor(ss)
            elif maxind2[i]*stride >= S_pre_sz[3] - w.shape[3]:
                ss = s_pre[0,:,maxind1[i] * stride:maxind1[i] * stride + w.shape[2], maxind2[i] * stride:]
                input[:,:, :ss.shape[2]] = torch.tensor(ss)
            elif maxind1[i]*stride >= S_pre_sz[2] - w.shape[2]:
                ss = s_pre[0,:,maxind1[i] * stride:, maxind2[i] * stride:maxind2[i] * stride + w.shape[3]]
                input[:,:ss.shape[1], : ] = torch.tensor(ss)
            else:
                input = s_pre[0,:,maxind1[i] * stride:maxind1[i]*stride+w.shape[2], maxind2[i]*stride:maxind2[i]*stride+w.shape[3]]
            # In the paper, it assume that neuron spike either before or after the post-neuron. So, if the 
                #neuron did input 1 that means it will LTD.
            #print("input size",input.shape)
            #print("w size",w[i].shape)
            input = torch.tensor(input)
            dw = input * a_minus * w[i, :, :] * (1 - w[i, :, :]) + \
                input * a_plus * w[i, :, :] * (1 - w[i, :, :]) - \
                a_minus * w[i, :, :] * (1 - w[i, :, :])

            w[i,:] = w[i,:]+dw
    
    return torch.tensor(w)