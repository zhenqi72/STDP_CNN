from STDP_selfmade import STDP_learning,get_update_index
import torch
import numpy as np

torch.set_printoptions(threshold=torch.tensor(10000))
def main():

    #mask2 is (1, 20, 3, 3)
    #s_pre is (1, 4, 6, 6)
    #s_cur is (1, 20, 3, 3)
    mask1  = np.load("mask1.npy")
    mask1 = torch.tensor(mask1)
    s_pre  = np.load("z1.npy")
    s_cur  = np.load("z2.npy")
    
    w   = np.load("w1.npy")
    w   = torch.tensor(w)
    v1  = np.load("v1.npy")
    v1  = torch.from_numpy(v1)
    
    print("mask1 is",mask1)
    print("s_pre is",s_pre)
    print("s_cur is",s_cur)
    print("s_cur fired position",np.where(s_cur != 0))
    #print("w is",w)
    #print("v5 is",v5)

    
    maxvel,maxind1,maxind2 = get_update_index(v1,mask1)#get the neuron in current plane to update weight
    print("maxind1",maxind1)
    print("maxind2",maxind2)
    maxvel = torch.squeeze(maxvel,0)
    S_pre_sz = s_pre.shape

    w_after = STDP_learning(S_pre_sz=S_pre_sz,s_pre=s_pre, s_cur=s_cur, w=w, threshold=1,  # Input arrays
                    maxval=maxvel, maxind1=maxind1, maxind2=maxind2,  # Indices
                    stride=2, a_minus=0.01, a_plus=0.01)
    #print("w after",w)
    #np.savetxt("w after.cvs",np.array(w))
    #np.savetxt("s_pre.cvs",np.array(s_pre))
    #np.savetxt("s_cur.cvs",np.array(s_cur))
    not_equal_positions = torch.where(w_after != w)
    print(not_equal_positions[0],not_equal_positions[1])

if __name__ == "__main__":
    main()