from STDP_selfmade import STDP_learning,get_update_index
import torch
import numpy as np

torch.set_printoptions(threshold=torch.tensor(10000))
def main():

    #mask2 is (1, 20, 3, 3)
    #s_pre is (1, 4, 6, 6)
    #s_cur is (1, 20, 3, 3)
    mask2  = np.load("mask2.npy")
    s_pre  = np.load("z3.npy")
    s_cur  = np.load("z5.npy")
    
    w  = np.load("w2.npy")
    w  = torch.tensor(w)
    v5  = np.load("v5.npy")
    v5 = torch.from_numpy(v5)
    
    print("mask2 is",mask2)
    print("s_pre is",s_pre)
    print("s_cur is",s_cur)
    #print("w is",w)
    #print("v5 is",v5)
    
    maxvel,maxind1,maxind2 = get_update_index(v5,mask2)#get the neuron in current plane to update weight
    print("maxind1",maxind1)
    print("maxind2",maxind2)
    maxvel = torch.squeeze(maxvel,0)
    S_pre_sz = s_pre.shape

    w_after = STDP_learning(S_pre_sz=S_pre_sz,s_pre=s_pre, s_cur=s_cur, w=w, threshold=10,  # Input arrays
                    maxval=maxvel, maxind1=maxind1, maxind2=maxind2,  # Indices
                    stride=2, a_minus=0.01, a_plus=0.01)
    #print("w after",w)
    #np.savetxt("w after.cvs",np.array(w))
    #np.savetxt("s_pre.cvs",np.array(s_pre))
    #np.savetxt("s_cur.cvs",np.array(s_cur))
    not_equal_positions = torch.where(w_after != w)
    print("不同元素的位置:", not_equal_positions)

if __name__ == "__main__":
    main()