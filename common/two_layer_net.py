import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.functions import *
from common.gradient import *

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy(y, t)

        return loss

    def accuracy(self, x, t):
        y_percent = self.forward(x)
        y_pred = np.argmax(y_percent, axis=1)
        accuracy = np.sum(y_pred == t) / x.shape[0]

        return accuracy

    def numerical_gradient(self, x, t):
        loss_func = lambda _ : self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_func, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_func, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_func, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_func, self.params['b2'])

        return grads


