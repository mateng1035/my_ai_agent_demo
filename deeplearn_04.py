import numpy as np
import torch
from torch.utils.data import  TensorDataset
from torch.utils.data import  DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import sys
import os

if sys.platform == "win32":
    # os.chdir(r'D:\MyProject\my_ai_agent_demo\my_ai_agent_demo\mt_model_train')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # os.chdir('/Users/mateng/硬盘/我的项目/Python/clould/my_ai_agent_demo/mt_model_train')

# numpy 对象 .> 张量Tensor -> 数据集对象TensorDataset -> 数据加载器DataLoader

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
    # dataset 数据集对象， batch_size 批次大小，shuffle 是否随机打乱数据
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 创建初始的线性回归模型
    model = nn.Linear(1, 1)

    # 创建损失函数
    criterion = nn.MSELoss()

    # 创建优化器对象
    # 模型参数，lr学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 具体的训练过程
    # 定义变量，分别表示： 训练轮数 100，每轮的（平均）损失值，训练总损失值，训练的样本树数
    epochs, lost_list, total_loss, total_sample = 100, [], 0.0, 0

    # 开始训练，按轮训练
    for epoch in range(epochs):
        # 每轮分批次训练， 从数据加载器中获取批次数据
        for train_x, train_y in dataloader:
            # 模型预测
            y_pred = model(train_x)
            # 计算每轮的损失
            loss = criterion(y_pred, train_y.reshape(-1, 1))  # train_y.reshape(-1, 1) 转为n行1列
            # 计算总损失 和 样本（批次）数
            total_loss += loss.item() # 标量取值
            total_sample += 1
            # 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad() # 梯度清零
            loss.backward()       # 反向传播
            optimizer.step()      # 梯度更新
        lost_list.append(total_loss / total_sample)
        print(f'轮数: {epoch + 1}, 平均损失：{total_loss / total_sample}')
    # 打印最终训练结果
    print(f'{epochs}论的平均损失为： {lost_list}')
    print(f'模型参数， 权重：{model.weight}， 偏置：{model.bias}')

    # 绘制曲线
    # plt.plot( range(epochs), lost_list)
    # plt.title('损失值曲线变化图')
    # plt.grid()
    # plt.show()

    # 绘制预测值和真实值的关系
    plt.scatter(x, y)
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])  # 偏置在creat_dateset的定义
    plt.plot(x, y_pred, color='red', label='预测值')
    plt.plot(x, y_true, color='blue', label='真实值')
    plt.legend()
    plt.grid()
    plt.show()

    # 总结：人工智能 = 数据 + 算法 + 算力

if __name__ == '__main__':
    x, y, coef = creat_dateset()
    # coef w的初始值
    # print(coef)
    train(x, y, coef)