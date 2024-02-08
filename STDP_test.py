from STDP_selfmade import STDP_learning,get_update_index,delta_t
import torch
import numpy as np

mask1  = np.load("mask1.npy")
mask2  = np.load("mask2.npy")
s2  = np.load("s2.npy")
s5  = np.load("s5.npy")
w1  = np.load("w1.npy")
w2  = np.load("w2.npy")
z2  = np.load("z2.npy")
z5  = np.load("z5.npy")
v2  = np.load("v2.npy")
v5  = np.load("v5.npy")




maxvel,index = get_update_index(v2,mask2)#get the neuron in current plane to update weight
maxind1 = index[0,:,0]
maxind2 = index[0,:,1]

w = STDP_learning(S_pre_sz=s2.shape,s_cur=s5,w=w,threshold=10,maxval=maxvel,maxind1=maxind1,maxind2=maxind2,stride=1,mask_pre_lay=mask_pre,a_minus=0.1,a_plus=0.1)