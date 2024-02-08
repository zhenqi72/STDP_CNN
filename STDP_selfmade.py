import numpy as np
import torch

def get_update_index(v,mask):
    v = v*mask
    maxvel,index = torch.max(input=v,dim=1)

    return maxvel,index
def check_previous_state(maxind1,maxind2,mask,stride):
    x_pre = maxind1 *stride
    y_pre = maxind2 *stride
    if mask[x_pre,y_pre] ==1:
        fired = 1
    else :
        fired = -1
    return fired

def STDP_learning(S_sz, s, w, K_STDP,  # Input arrays
                  maxval, maxind1, maxind2,  # Indices
                  stride, mask_pre_lay, a_minus, a_plus):  # Parameters

    for i in range(w.shape[0]):
        fired=check_previous_state(maxind1[i],maxind2[i],mask_pre_lay,stride)
        if delta_t > 0:
            dw = a_plus*w*(1-w)
        else:
            dw = a_minus*w*(1-w)

        w[i,:] = w[i,:]+dw

        # Weights STDP update
        

    return w, K_STDP