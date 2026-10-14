import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.optimizer import *

# 定义目标函数
def f(x, y):
    return x ** 2 / 20  + y ** 2

# 定义梯度计算方法
def f_grad(x, y):
    return x / 10, 2 * y

init_pos = (-7.0, 2.0)
params = {}
grads = {}

# 定义优化器
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.9)
optimizers['Momentum'] = Momentum(lr=0.08, momentum=0.85)
optimizers['AdaGrad'] = AdaGrad(lr=1)
optimizers['Adam'] = Adam(lr=0.3, beta1=0.5)

# 画图序号
idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history, y_history = [], []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(50):
        # 保存当前点的坐标
        x_history.append(params['x'])
        y_history.append(params['y'])

        # 计算梯度
        grads['x'], grads['y'] = f_grad(params['x'], params['y'])

        # 更新参数
        optimizer.update(params, grads)

    # 画图
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    Z[Z > 7] = 0

    plt.subplot(2, 2, idx)
    idx += 1
    plt.contour(X, Y, Z)
    plt.plot(0, 0, '+')
    plt.plot(x_history, y_history, '-', color='red', label=key)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.legend(loc='best')

plt.show()