import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.functions import *
from common.gradient import *
from common.two_layer_net import *
from common.load_data import *

x_train, x_test, t_train, t_test = get_data()

model = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

lr = 0.1
batch_size = 100
epochs = 10
train_size = x_train.shape[0]
iters_per_epoch = int(np.ceil(train_size / batch_size))
train_loss = []
train_acc = []
test_acc = []

for i in range(epochs):
    for _ in range(iters_per_epoch):
        # 随机选取批量数据
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        print(f"Epoch {i + 1}, batch {_ + 1}/{iters_per_epoch}: 开始计算梯度")
        grad = model.numerical_gradient(x_batch, t_batch)
        print("梯度计算完成")

        # 更新参数
        for key in model.params.keys():
            model.params[key] -= grad[key] * lr

        # 计算并保存当前训练损失
        loss = model.loss(x_batch, t_batch)
        train_loss.append(loss)

    train_acc.append(model.accuracy(x_train, t_train))
    test_acc.append(model.accuracy(x_test, t_test))
    print(f"Epoch:{i}, Train_acc:{train_acc[i]}, Test_acc:{test_acc[i]}")

x = np.arange(len(train_acc))
plt.plot(x, train_acc, label='Train Acc')
plt.plot(x, test_acc, label='Test Acc',linestyle='--')
plt.legend(loc='best')
plt.show()