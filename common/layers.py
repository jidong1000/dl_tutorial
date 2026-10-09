import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.functions import *
import numpy as np

class Relu():
    def __init__(self):
        # 记录哪些输入小于等于0
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        # 将x<=0的位置都赋值为0
        y[self.mask] = 0

        return y

    def backward(self, dy):
        dx = dy.copy()
        dx[self.mask] = 0

        return dx

class Sigmoid():
    def __init__(self):
        # 记录输出值y，用于反向传播时计算梯度
        self.y = None

    def forward(self, x):
        y = sigmoid(x)
        self.y = y

        return y

    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y

        return dx

# Affine, 仿射变换, 用于全连接层
# y = x * w + b
class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        y = np.dot(self.x, self.W) + self.b
        
        return y

    def backward(self, dy):
        dx = np.dot(dy, self.W.T)
        dx = dx.reshape(*self.original_x_shape)
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)

        return dx
    
# 输出层
class SoftmaxWithLoss():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)

        return self.loss

    def backward(self, dy=1):
        n = self.t.shape[0]
        # 独热编码
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 顺序编码
        else:
            dx = self.y.copy()
            # self.t如果是顺序编码, 即为正确标签, 也是每条数据预测值所在列号
            dx[np.arange(n), self.t] -= 1

        return dx / n