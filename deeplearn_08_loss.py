import torch
from torch import nn

def test01():
    # 多任务损失函数
    # 多分类的结果，注意1行只有1个1
    y_true = torch.tensor([[0,1,0],[0,0,1]], dtype=torch.float32)
    # 多分类问题，注意 行相加为1
    y_pred = torch.tensor([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]], requires_grad=True, dtype=torch.float32)

    loss = nn.CrossEntropyLoss()

    my_loss = loss(y_pred, y_true).detach().numpy()
    print(f'loss {my_loss}')

def test02():
    # 二分类任务损失函数
    # loss = -ylog(预测值) - (1-y)log(1-预测值)
    y_pred = torch.tensor([0.6901, 0.5459, 0.2469], requires_grad=True)
    y_true = torch.tensor([0, 1, 0], dtype=torch.float32)
    loss = nn.BCELoss()
    my_loss = loss(y_pred, y_true).detach().numpy()
    print(f'loss {my_loss}')


def test03():
    # 回归任务损失函数-MAE损失函数
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    y_true = torch.tensor([2, 2, 2], dtype=torch.float32)
    # loss = (1/n)(Σ|真实值-预测值|)
    # ((2-1) + (2-1) + (2-1.9))/3 = 2.1 / 3 = 0.7
    loss = nn.L1Loss()
    my_loss = loss(y_pred, y_true).detach().numpy()
    print(f'loss {my_loss}')

def test04():
    # 回归任务损失函数-MSE损失函数
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    y_true = torch.tensor([2, 2, 2], dtype=torch.float32)
    # loss = (1/n)(Σ|真实值-预测值|)
    # ((2-1)^2 + (2-1)^2 + (2-1.9)^2)/3 = 2.01 / 3 = 0.67
    loss = nn.MSELoss()
    my_loss = loss(y_pred, y_true).detach().numpy()
    print(f'loss {my_loss}')

def test05():
    # 回归任务损失函数-Smooth L1函数
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    y_true = torch.tensor([2, 2, 2], dtype=torch.float32)

    loss = nn.SmoothL1Loss()
    my_loss = loss(y_pred, y_true).detach().numpy()
    print(f'loss {my_loss}')

if __name__ == '__main__':
    test05()