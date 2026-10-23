import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# 构建数据集
x = torch.randn(100, 1)
w = torch.tensor([2.5])
b = torch.tensor([5.2])
noise = torch.randn(100, 1) * 0.5

y = w * x + b + noise

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 构建模型
model = nn.Linear(in_features=1, out_features=1)

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 模型训练
epoches = 200
loss_list = []
for epoch in range(epoches):
    total_loss = 0
    for x_train, y_train in dataloader:
        # 前向传播
        y_pred = model(x_train)
        # 计算损失
        loss_val = loss(y_pred, y_train)
        total_loss += loss_val.item()
        # 反向传播
        loss_val.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
    # 记录每一个epoch的平均损失值
    loss_list.append(total_loss / len(dataset))

# 打印训练完的模型参数
print(model.weight)
print(model.bias)

# 画图
fig, ax = plt.subplots(1, 2)
ax[0].plot(loss_list)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')

ax[1].scatter(x, y)
y_pred = model.weight.item() * x + model.bias.item()
ax[1].plot(x, y_pred, color='red')
plt.show()
