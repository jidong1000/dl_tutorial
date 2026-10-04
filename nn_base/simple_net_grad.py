import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common.functions import *
from common.gradient import *

class SimpleNet():
    # 初始化网络
    def __init__(self):
        self.W = np.random.randn(2, 3)

    # 前向传播
    def forward(self, x):
        a = np.dot(x, self.W)
        y = softmax(a)

        return y

    # 计算损失值
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy(y, t)

        return loss

if __name__ == "__main__":
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    network = SimpleNet()
    # 因为lambda表达式会得到一个funcion对象，这个是numerical_gradient的第一个参数需要的类型
    f = lambda _: network.loss(x, t)
    dw = numerical_gradient(f, network.W)
    print(dw)