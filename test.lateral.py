import torch
from Lateral_inhibit import Later_inhibt
def main():
    torch.set_printoptions(edgeitems=3, threshold=10000)
    lateral = Later_inhibt([1,3,5,3])
    spikes1 = torch.zeros(1,3,5,3)
    spikes2 = torch.zeros(1,3,5,3)
    spikes3 = torch.zeros(1,3,5,3)
    row_index1 = torch.randint(0,4,(4,))
    col_index1 = torch.randint(0,2,(4,))
    row_index2 = torch.randint(0,4,(4,))
    col_index2 = torch.randint(0,2,(4,))

    spikes1[0,1,row_index1,col_index1] = 1 
    spikes2[0,2,row_index2,col_index2] = 1 
    spikes3[0,1,row_index2,col_index2] = 1
    print("spikes1",spikes1)
    print("spikes2",spikes2)
    print("spikes3",spikes3)
    mask1  = torch.ones(1,3,5,3)
    result1,mask1 = lateral(spikes1,mask1,0)
    
    result2,mask2 = lateral(spikes2,mask1,1)
    
    result3,mask3 = lateral(spikes3,mask2,1)
    print("mask1",mask1)
    print("mask2",mask2)
    print("mask3",mask3)
    print("result1 is",result1)
    print("result2 is",result2)
    print("result3 is",result3)
if __name__ == "__main__":
    main()