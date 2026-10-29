import torch
import torch.nn as nn
from torchsummary import summary

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义神经网络中的每一层
        self.linear1 = nn.Linear(3, 4)
        nn.init.xavier_normal_(self.linear1.weight)
        self.tanh = nn.Tanh()

        self.linear2 = nn.Linear(4, 4)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.relu = nn.ReLU()

        self.linear3 = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)
        y = self.softmax(x)

        return y

# 统一模型和输入所在的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集
x = torch.randn(10, 3, device=device)

# 创建神经网络模型
model = Model().to(device)
# 查看模型权重参数
for name, param in model.named_parameters():
    print(f"{name} {param}")
# 查看模型的结构
summary(model, (3,), 10, device=device.type)

# 前向传播
output = model(x)

print(output)
