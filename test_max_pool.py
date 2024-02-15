import torch
from Max_pooling_snn import Max_pool_snn
# 定义 Max_pool_snn 类
# 初始化 Max_pool_snn 实例
def main():
    kernel_size = 3
    stride = 2
    padding = 0
    pooling_layer = Max_pool_snn(kernel_size, stride, padding)

    # 创建一个随机的输入张量 x 和初始掩码张量 mask
    x = torch.zeros(size=(1,12,12))  # 假设输入是一个1x1x6x6的张量
    row_index1 = torch.randint(0,12,(10,))
    col_index1 = torch.randint(0,12,(10,))
    x[0,row_index1,col_index1] = 1
    mask = torch.ones(1,5,5)   # 创建与 pooling 以后形状相同、全为1的掩码张量
    x2 = x

    # 调用 Max_pool_snn 的 forward 方法
    print("x is ",x)
    o_resul1,result, mask = pooling_layer.forward(x, mask)
    print("o_resul1 is",o_resul1)
    print("x2 is",x2)
    index = torch.where(x2 == 0)
    x2[index] = 1 

    # 打印结果
    print("Output after max pooling:", result)
    print("Updated mask:", mask)

    o_resul2,result2, mask2 = pooling_layer.forward(x2, mask)
    print("o_resul2 is",o_resul2)
    print("Output2 after max pooling:", result2)
    print("Updated2 mask:", mask2)

if __name__ == "__main__":
    main()
