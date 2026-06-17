# import numpy as np
# import torch
#
# data = torch.ones((3,3)) # 创建3x3的全为1的向量
# print(data) # 输出创建
# data1 = torch.tensor(data, dtype=torch.int) # 转换为int类型 方式1
# print(data1)
# data2 = data.int() # 转换为int类型 方式2
# print(data2)



# import torch
#
# # 方式 1 cuda
# x = torch.tensor([1, 2, 3])
#
# x_gpu1 = x.cuda()
#
# print(x_gpu1.device)
#
# # 方式 2 to
#
# x_gpu2 = x.to("cuda")
#
# print(x_gpu2.device)
#
# # 方式 3 直接指定设备
#
# x = torch.tensor(
#     [1, 2, 3],
#     device="cuda"
# )
#
# print(x.device)
#
# # 方式 4 使用 torch.device
#
# device = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "cpu"
# )
#
# x = torch.tensor([1, 2, 3])
#
# x = x.to(device)
#
# print(x.device)

# import torch
# import numpy as np
#
# data = torch.full(size=(4, 3), fill_value=2)
# print('创建(4, 3)的张量, ', data)
# array = data.numpy().copy()
# print('转换为Numpy数组 ', array)

# 创建一个形状为（5,4）的随机张量，取出第二到第四行，第一到第三列之间的元素

# import torch
#
# data = torch.rand(5,4)
# print('创建(5,4)的张量, ', data)
# newdata = data[1: 4, 0: 3]
# print('取出第二到第四行，第一到第三列之间的元素', newdata)


# 创建一个形状为（3,4,5）的随机张量，将其形状变为（4,3,5）
# import torch
# data = torch.rand((3, 4, 5))
# print('创建(3, 4, 5)的张量，其形状shape是, ', data.shape)
# new_data = data.reshape((4, 3, 5))
# print('将其形状变为（4,3,5），转换后的形状shape是, ', new_data.shape)

import numpy as np
import torch
from torch.utils.data import  TensorDataset
from torch.utils.data import  DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression

def creat_dateset():
    x, y, coef = make_regression(
        n_samples=100,  # 100条样本
        n_features=1,   # 1个特征
        noise=10,       # 噪声，噪声越大，样本点越散，噪声越小，样本点越集中
        coef=True,      # 是否返回系数，默认为False，返回值为None
        bias=14.5,      # 偏置
        random_state=3  # 随机种子，随机种子相同，输出数据相同
    )
    # x y 封装为张量对象
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    return x, y, coef

def train(x, y, coef):
    # 创建数据集对象，把 tensor -> 数据集对象 -> 数据加载器
    dataset = TensorDataset(x, y)

    # 创建数据加载对象
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 创建初始的线性回归模型
    model = nn.Linear(1, 1)

    # 创建损失函数
    criterion = nn.MSELoss()

    # 创建优化器对象
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 具体的训练过程
    # 定义变量，分别表示： 训练轮数 100，每轮的（平均）损失值，训练总损失值，训练的样本树数
    epochs, lost_list, total_loss, total_sample = 100, [], 0.0, 0

    # 开始训练，按轮训练
    for epoch in range(epochs):
        for train_x, train_y in dataloader:
            # 模型预测
            y_pred = model(train_x)
            # 计算每轮的损失
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            # 计算总损失 和 样本（批次）数
            total_loss += loss.item() # 标量取值
            total_sample += 1
            # 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lost_list.append(total_loss / total_sample)
        print(f'轮数: {epoch + 1}, 平均损失：{total_loss / total_sample}')
    # 打印最终训练结果
    print(f'{epochs}论的平均损失为： {lost_list}')
    print(f'模型参数， 权重：{model.weight}， 偏置：{model.bias}')

if __name__ == '__main__':
    x, y, coef = creat_dateset()
    train(x, y, coef)