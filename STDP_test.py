from STDP_selfmade import STDP_learning,get_update_index,delta_t
import torch
import numpy as np

torch.set_printoptions(threshold=torch.inf)
def main():
    mask2  = np.load("mask2.npy")
    s_pre  = np.load("z3.npy")
    s_cur  = np.load("z5.npy")
    w  = np.load("w2.npy")
    v5  = np.load("v5.npy")

    print("mask2 is",mask2)
    print("s_pre is",s_pre)
    print("s_cur is",s_cur)
    print("w is",w)
    print("v5 is",v5)



    v5 = torch.from_numpy(v5)
    maxvel,index = get_update_index(v5,mask2)#get the neuron in current plane to update weight
    maxind1 = index[0,:,0]
    maxind2 = index[0,:,1]
    S_pre_sz = s_pre.shape

    w = STDP_learning(S_pre_sz=S_pre_sz,s_pre=s_pre, s_cur=s_cur, w=w, threshold=10,  # Input arrays
                    maxval=maxvel, maxind1=maxind1, maxind2=maxind2,  # Indices
                    stride=1, a_minus=0.01, a_plus=0.01)

if __name__ == "__main__":
    main()