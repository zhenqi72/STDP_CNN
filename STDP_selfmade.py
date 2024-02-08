import numpy as np
import torch

def get_update_index(v,mask):
    v = v*mask #get the neuron that can fire
    maxvel,index = torch.max(input=v,dim=1)

    return maxvel,index

def delta_t(fired,ss):
    d_t = fired*ss
    return d_t

def STDP_learning(S_pre_sz,s_pre, s_cur, w, threshold,  # Input arrays
                  maxval, maxind1, maxind2,  # Indices
                  stride, mask_pre_lay, a_minus, a_plus):  # Parameters
    
    for i in range(w.shape[0]):
        if maxval[i]>threshold:
            ss=s_cur[0,i,maxind1[i],maxind2[i]]
            if maxind2[i]*stride >= S_pre_sz[3] - w.shape[2] and maxind1[i]*stride >= S_pre_sz[2] - w.shape[1]:
                fired = mask_pre_lay[0,i,maxind1[i] * stride:, maxind2[i] * stride: ]
            elif maxind2[i]*stride >= S_pre_sz[1] - w.shape[2]:
                fired = mask_pre_lay[0,i,maxind1[i] * stride:maxind1[i] * stride + w.shape[0], maxind2[i] * stride:]
            elif maxind1[i]*stride >= S_pre_sz[0] - w.shape[0]:
                fired = mask_pre_lay[0,i,maxind1[i] * stride:, maxind2[i] * stride:maxind2[i] * stride + w.shape[2]]
            else:
                fired = mask_pre_lay[0,i,maxind1[i] * stride:maxind1[i]*stride+w.shape[1], maxind2[i]*stride:maxind2[i]*stride+w.shape[2]]
                
            d_t = delta_t(fired,ss)
            dw = d_t * a_minus * w[i, :, :] * (1 - w[i, :, :]) + \
                d_t * a_plus * w[i, :, :] * (1 - w[i, :, :]) - \
                a_minus * w[i, :, :] * (1 - w[i, :, :])

            w[i,:] = w[i,:]+dw
    
    return w