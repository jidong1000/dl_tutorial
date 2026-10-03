import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common.functions import *

def network():
    network = {}
    # 第一层
    network['W1'] = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    # 第二层
    network['W2'] = np.array([
        [0.1, 0.3],
        [0.2, 0.4],
        [0.5, 0.6]
    ])
    network['B2'] = np.array([0.1, 0.2])
    # 第三层
    network['W3'] = np.array([
        [0.7, 0.8],
        [0.2, 0.6]
    ])
    network['B3'] = np.array([0.4, 0.3])

    return network

def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    Y = identity(a3)

    return Y

network = network()
X = np.array([1, 2])
X_pred = forward(network, X)
print(X_pred)

    