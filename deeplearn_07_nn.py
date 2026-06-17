import torch
from torch import nn
from torchsummary import summary

class ModelDemo(nn.Module):
    # 完成初始化父类成员，及神经网络
    def __init__(self):
        super().__init__()
        # 搭建神经网络 -> 隐藏层 + 输出层
        # 隐藏层1
        self.linner1 = nn.Linear(3, 3)
        # 隐藏层2
        self.linner2 = nn.Linear(3, 2)
        # 输出层
        self.output = nn.Linear(2, 2)

        # 参数初始化
        # 隐藏层1
        nn.init.xavier_normal_(self.linner1.weight)
        nn.init.zeros_(self.linner1.bias)

        # 隐藏层2
        nn.init.kaiming_normal_(self.linner2.weight)
        nn.init.zeros_(self.linner2.bias)



    def forward(self, x):
        # 第一层 隐藏层计算： 加权求和 + 激活函数
        # 分解版写法
        # x = self.linner1(x)  # 加权求和
        # x = torch.sigmoid(x) # 激活函数
        # 合并版写法
        x = torch.sigmoid(self.linner1(x))
        # 第二层
        x = torch.relu(self.linner2(x))
        # 输出层 dim=-1表示按行计算，一条样本一个处理
        x = torch.softmax(self.output(x), dim=-1)
        # 返回预测值
        return x

def train():
    # 创建模型对象
    model = ModelDemo()
    # 创建数据集样本
    data = torch.randn(5, 3)
    print(f'data: {data}')
    print(f'data shape: {data.shape}')
    print(f'data requires_grad: {data.requires_grad}') # 是否自动微分
    # 调用神经网络模型，进行模型训练
    output = model(data)  # 底层自动调用了forward方法，进行前向传播
    print(f'output: {output}')
    print(f'output shape: {output.shape}')
    print(f'output requires_grad: {output.requires_grad}') # 是否自动微分

    # 计算和查看模型
    summary(model, input_size=(5, 3)) # 列必须是3，行可以随意
    print('查看参数w和b')
    for name, paramter in model.named_parameters():
        print(f'name: {name}, paramter: {paramter}')


if __name__ == '__main__':
    train()