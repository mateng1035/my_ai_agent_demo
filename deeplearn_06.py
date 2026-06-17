import torch
import numpy as np
from torch import nn

def param_init(type:int):
    # 创建一个线性层 输入5个神经元，输出3个神经元
    linear = nn.Linear(5, 3)
    if type == 1:
        nn.init.uniform_(linear.weight)  # 从0-1均匀分布产生参数
        nn.init.uniform_(linear.bias) # 偏置初始化
    elif type == 2:
        nn.init.constant_(linear.weight, 5) # 固定初始化
        nn.init.constant_(linear.bias, 5)
    elif type == 3:
        nn.init.zeros_(linear.weight) # 全0初始化
        nn.init.zeros_(linear.bias)
    elif type == 4:
        nn.init.ones_(linear.weight) # 全1初始化
        nn.init.ones_(linear.bias)
    elif type == 5:
        nn.init.normal_(linear.weight, mean=0, std=1) # 正态分布随机初始化
        nn.init.normal_(linear.bias, mean=0, std=1)
    elif type == 6:
        nn.init.kaiming_normal_(linear.weight) # kaiming 正态分布初始化
        nn.init.kaiming_normal_(linear.bias)
    elif type == 7:
        nn.init.kaiming_uniform_(linear.weight) # kaiming 均匀分布初始化
        nn.init.kaiming_uniform_(linear.bias)
    elif type == 8:
        nn.init.xavier_uniform_(linear.weight) # xavier 正态分布初始化
        nn.init.xavier_uniform_(linear.bias)
    elif type == 9:
        nn.init.xavier_normal_(linear.weight) # xavier 均匀分布初始化
        nn.init.xavier_normal_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)

if __name__ == '__main__':
    param_init(3)