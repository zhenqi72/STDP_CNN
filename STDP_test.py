from STDP_selfmade import STDP_learning,get_update_index,delta_t
import torch
import numpy as np

torch.set_printoptions(threshold=torch.tensor(10000))
def main():
    mask2  = np.load("mask2.npy")
    mask2  = mask2[:,:,0:3,0:3]
    s_pre  = np.load("z3.npy")
    s_pre  = s_pre[:,:,0:6,0:6]
    s_cur  = np.load("z5.npy")
    s_cur  = s_cur[:,:,0:3,0:3]
    w  = np.load("w2.npy")
    w = w[:,:,0:3,0:3]
    v5  = np.load("v5.npy")
    v5 = torch.from_numpy(v5)
    v5 = v5[:,:,0:3,0:3]

    print("mask2 is",mask2)
    print("s_pre is",s_pre.shape)
    print("s_cur is",s_cur.shape)
    print("w is",w.shape)
    print("v5 is",v5.shape)



    
    maxvel,maxind1,maxind2 = get_update_index(v5,mask2)#get the neuron in current plane to update weight
    maxvel = torch.squeeze(maxvel,0)
    S_pre_sz = s_pre.shape

    w = STDP_learning(S_pre_sz=S_pre_sz,s_pre=s_pre, s_cur=s_cur, w=w, threshold=10,  # Input arrays
                    maxval=maxvel, maxind1=maxind1, maxind2=maxind2,  # Indices
                    stride=2, a_minus=0.01, a_plus=0.01)
    
    np.savetxt("w after.cvs",np.array(w))
    np.savetxt("s_pre.cvs",np.array(s_pre))
    np.savetxt("s_cur.cvs",np.array(s_cur))

if __name__ == "__main__":
    main()