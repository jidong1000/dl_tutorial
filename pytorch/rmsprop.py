import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD, RMSprop

# 定义函数
def f(x):
    return 0.05 * (x[0] ** 2) + x[1] ** 2

# 实现梯度下降法
def gradient_descent(x, optimizer, num_iters):
    # 拷贝当前x的值，放入列表中
    x_arr = x.detach().numpy().copy()
    for i in range(num_iters):
        # 前向传播
        y = f(x)
        # 反向传播
        y.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        x_arr = np.vstack([x_arr, x.detach().numpy()])

    return x_arr

x = torch.tensor([-7.0, 2.0], requires_grad=True)

lr = 0.1
num_iters = 90

# SGD
x_clone = x.clone().detach().requires_grad_(True)
optimizer1 = SGD([x_clone], lr=lr)
x_arr1 = gradient_descent(x_clone, optimizer1, num_iters)
plt.plot(x_arr1[:, 0], x_arr1[:, 1], color='b')

# RMS
x_clone = x.clone().detach().requires_grad_(True)
optimizer2 = RMSprop([x_clone], lr=lr, alpha=0.9)
x_arr2 = gradient_descent(x_clone, optimizer2, num_iters)
plt.plot(x_arr2[:, 0], x_arr2[:, 1], color='r')

plt.show()