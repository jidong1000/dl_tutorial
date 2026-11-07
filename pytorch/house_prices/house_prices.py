import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

data_pwd = r'D:\project\deep_learning\dl_tutorial\data\house_prices.csv'

# 创建数据集
def load_data(pwd):
    data = pd.read_csv(pwd)
    # 去除无关列
    data.drop(['Id'], axis=1, inplace=True)
    # 划分特征和目标
    x = data.drop(['SalePrice'], axis=1)
    y = data['SalePrice']
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    # 按照特征数据类型划分成数值型和类别型
    numerical_features = x.select_dtypes(exclude=['string']).columns
    categorical_features = x.select_dtypes(include=['string']).columns
    # 用平均值填充数值项缺失，再标准化
    numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('std', StandardScaler())
        ]
    )
    # 用NaN填充分类项缺失，再独热编码
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    # 组合列转换器
    transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features), 
            ('cat', categorical_transformer, categorical_features),

        ]
    )
    # 把预处理的规则套用到训练集和测试集上
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)
    x_train = pd.DataFrame(x_train.toarray(), columns=transformer.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=transformer.get_feature_names_out())
    # 创建数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))
    # 还需要返回输入的特征数量
    return train_dataset, test_dataset, x_train.shape[1]

train_dataset, test_dataset, feature_num = load_data(data_pwd)

# 搭建模型
model = nn.Sequential(
    nn.Linear(feature_num, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
)

# 定义损失函数
def log_rmse(y_pred, target):
    y_pred = torch.clamp(y_pred, 1, float('inf'))
    mse = nn.MSELoss()
    return torch.sqrt(mse(torch.log(y_pred), torch.log(target)))

# 模型训练测试
def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    # 初始化
    def init_params(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
    model.apply(init_params)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    test_loss_list = []

    # 主流程
    for epoch in range(epoch_num):
        # 训练
        model.train()
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_loss_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 前向传播
            y_pred = model(x)
            # 计算损失
            loss_value = log_rmse(y_pred.squeeze(-1), y)
            # 反向传播
            loss_value.backward()
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            train_loss_total += loss_value.item() * x.shape[0]

        current_train_loss = train_loss_total / len(train_dataset)
        train_loss_list.append(current_train_loss)

        # 测试
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        test_loss_total = 0
        # 测试时关闭梯度计算
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss_value = log_rmse(y_pred.squeeze(-1), y)

                test_loss_total += loss_value * x.shape[0]

        current_test_loss = test_loss_total.item() / len(test_dataset)
        test_loss_list.append(current_test_loss)

        print(f"epoch:{epoch + 1}, train loss:{current_train_loss}, test loss:{current_test_loss}")
    
    return train_loss_list, test_loss_list

lr = 0.1
epoch_num = 200
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)

# 画图
plt.plot(train_loss_list, 'r-', label='train loss')
plt.plot(test_loss_list, 'b--', label='test loss')
plt.legend()
plt.show()
