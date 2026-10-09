import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.functions import *
# from common.gradient import *
from common.layers import *
from collections import OrderedDict

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        # 定义层结构
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def forward(self, x):
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']

        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)

        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)

        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.forward(x)
        # loss = cross_entropy(y, t)
        loss = self.lastLayer.forward(y, t)

        return loss

    def accuracy(self, x, t):
        y_percent = self.forward(x)
        y_pred = np.argmax(y_percent, axis=1)
        accuracy = np.sum(y_pred == t) / x.shape[0]

        return accuracy

    # def numerical_gradient(self, x, t):
    #     loss_func = lambda _ : self.loss(x, t)
    #     grads = {}
    #     grads['W1'] = numerical_gradient(loss_func, self.params['W1'])
    #     grads['b1'] = numerical_gradient(loss_func, self.params['b1'])
    #     grads['W2'] = numerical_gradient(loss_func, self.params['W2'])
    #     grads['b2'] = numerical_gradient(loss_func, self.params['b2'])

    #     return grads

    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t)

        # 反向传播
        dy = self.lastLayer.backward()

        # 将神经网络中的层进行反向处理
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dy = layer.backward(dy)

        # 提取各层参数的梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
     