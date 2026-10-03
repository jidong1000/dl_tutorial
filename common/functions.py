import numpy as np

# 激活函数

'''
隐藏层：
1.首选ReLU, 如果效果不好可以常识ReLU的变体
2.sigmoid在隐藏层中容易导致梯度消失, 应该尽量避免
3.tanh输出的均值为0, 对中心化数据更友好, 也可能梯度消失, 仅适用于浅层网络

输出层：
1.二分类选择sigmoid
2.多分类选择softmax
3.回归默认选择identify
'''

def step_function(x):
    return np.array(x >= 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    # 溢出处理
    if x.ndim == 2:
        return np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def identity(x):
    return x

# 损失函数
def mean_squared_error(y, t):
    return (np.sum(y - t) ** 2) * 0.5

def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    n = y.shape[0]

    return -np.sum(np.log(y[np.arange(n), t] + 1e-10)) / n

if __name__ == "__main__":
    x = np.array([0, 2, 3, 4, -1, -2])
    X = np.array([[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8],
                 [5, 6, 7]])