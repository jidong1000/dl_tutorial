import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.functions import *

def get_data():
    # 从csv文件中读取数据集
    data = pd.read_csv(r"D:\project\deep_learning\dl_tutorial\data\train.csv")
    # 划分数据集
    X = data.drop('label', axis=1)
    Y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    # 归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    # 训练集使用transform()是为了直接使用已经学习到的规则转换数据
    x_test = scaler.transform(x_test)

    return x_test, y_test

def forward(network, X):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(X, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    Y = softmax(a3)

    return Y

# 获取数据
X, Y = get_data()
print(X.shape, Y.shape)

# 加载网络
network = joblib.load(r"D:\project\deep_learning\dl_tutorial\data\nn_sample")

# 前向传播
Y_percent = forward(network, X)
print(Y_percent.shape)

# 将分类概率转化为分类标签
Y_pred = np.argmax(Y_percent, axis=1)
print(Y_pred.shape)

# 计算准确率
accuracy_cnt = np.sum(Y == Y_pred)
print(f"准确率:{accuracy_cnt / X.shape[0]}")
