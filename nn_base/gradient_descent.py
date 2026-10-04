import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.functions import *
from common.gradient import *

def gradient_descent(f, init_x, lr=0.01, iter=100):
    x = init_x
    x_history = []

    for _ in range(iter):
        x_history.append(x.copy())
        # 计算梯度
        grad = numerical_gradient(f, x)
        # 更新参数
        x -= lr * grad

    return x, np.array(x_history)

def f(x):
    return x[0] ** 2 + x[1] ** 2

if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    lr = 0.01
    iter = 500

    x, x_history = gradient_descent(f, init_x, lr, iter)
    print(f"最小值点为：{x}")

    plt.scatter(x_history[:, 0], x_history[:, 1])
    plt.show()

    