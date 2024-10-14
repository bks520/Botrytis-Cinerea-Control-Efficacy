import numpy as np
import pandas as pd
import transtab
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 载入波士顿房价数据集
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.Series(boston.target, name='MEDV')
df = pd.concat([data, target], axis=1)

# 准备特征和目标数据
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用 Transtab 进行对比学习和输出嵌入
model, _ = transtab.build_contrastive_learner([], range(X_train.shape[1]), [], supervised=True)
model.train()  # 进入训练模式

# 转换为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# 定义简单的 MLP 网络作为回归模型
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义模型和优化器
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 1
mlp_model = MLPRegressor(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

# 在测试集上进行预测
mlp_model.eval()  # 进入评估模式
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred_tensor = mlp_model(X_test_tensor)
y_pred = y_pred_tensor.numpy()

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('均方误差 (MSE):', mse)

# ... 前面的代码不变 ...

# 定义计算 R^2 的函数
def r_squared(y_true, y_pred):
    y_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_total
    return r2.item()

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        r2 = r_squared(y_train_tensor, outputs)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}, R^2: {r2}')

# 在测试集上进行预测
mlp_model.eval()  # 进入评估模式
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred_tensor = mlp_model(X_test_tensor)
y_pred = y_pred_tensor.numpy()

# 计算均方误差和 R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(torch.tensor(y_test.values, dtype=torch.float32), torch.tensor(y_pred, dtype=torch.float32))
print('均方误差 (MSE):', mse)
print('R^2:', r2)



