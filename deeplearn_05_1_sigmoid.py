import torch
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

_, axes = plt.subplots(1, 2)
x = torch.linspace(-20, 20, 1000)
y = torch.sigmoid(x)
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('Sigmoid 函数图像')

x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.sigmoid(x).sum().backward()
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('Sigmoid 导数图像')
plt.show()