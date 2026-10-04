import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data(pwd=r'D:\project\deep_learning\dl_tutorial\data\train.csv'):
    data = pd.read_csv(pwd)

    x = data.drop('label', axis=1)
    y = data['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # 归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    # 训练集使用transform()是为了直接使用已经学习到的规则转换数据
    x_test = scaler.transform(x_test)

    # 将数据都转成ndarray
    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test