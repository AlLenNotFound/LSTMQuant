import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


class StockPriceRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPriceRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的隐藏状态进行预测
        return out


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0:5]  # 选择前look_back天的'Open', 'High', 'Low', 'Close', 'Volume'
        X.append(a)
        Y.append(dataset[i + look_back, 4])  # 对应未来一天的'Close'价格
    return np.array(X), np.array(Y)


def create_sequences(data, seq_length):
    result = []
    for index in range(len(data) - seq_length):
        result.append(data[index: index + seq_length])
    return result


data = pd.read_csv('600233.SH.csv')
data = data[['trade_date', 'open', 'high', 'low', 'close', 'vol']]
data['DaysOfWeek'] = pd.to_datetime(data['trade_date']).dt.dayofweek
data['trade_date'] = pd.to_datetime(data['trade_date']).dt.dayofyear
data['cycle'] = data['trade_date'] * (2 * np.pi / 365)
data['cycle'] = np.sin(data['cycle'])

seq_length = 30
X = []
y = []
for seq in create_sequences(data.values, seq_length):
    X.append(seq[:, :])  # 特征：open, high, low, close, vol
    y.append(seq[:, 4][-1])  # 目标：下一天的close价格
X = np.array(X)
y = np.array(y)
# 数据标准化
X_reshaped = X.reshape((X.shape[0], -1))
scaler = MinMaxScaler(feature_range=(-10, 10))
X_scaled = scaler.fit_transform(X_reshaped)
# [samples, time steps, features]
num_features = 8
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], seq_length, num_features))
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 现在X_train, y_train可以用于训练LSTM模型
# X_test, y_test可以用于验证模型性能
model = StockPriceRegressor(input_size=num_features, hidden_size=32, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 16
k = 0
res = []
Loss = []
'''
for i in range(0, len(X_train)):
    x_t = []
    y_t = []
    k += 1
    if i + batch_size < len(X_train):
        x_t.append(X_train[i:i + batch_size, :, :])
        y_t.append(y[i + batch_size:i + 2 * batch_size])
        print(i, "  ", batch_size + i, " ", i + batch_size, "  ", i + 2 * batch_size, len(y))
    else:
        x_t.append(X_train[len(X_train)-batch_size:len(X_train), :, :])
        y_t.append(y[len(X_train)-1:len(X_train) + batch_size])
        print("last===========>", i, "  ", batch_size + i)
        break
    x_t = np.array(x_t)
    X_train_tensor = torch.from_numpy(x_t).float()
    X_train_tensor = X_train_tensor.to(device)
    X_train_tensor = X_train_tensor.squeeze(0)
    y_t = np.array(y_t)
    Y_train_tensor = torch.from_numpy(y_t).float()
    Y_train_tensor = Y_train_tensor.to(device)
    Y_train_tensor = Y_train_tensor.permute(1, 0)
    print("===============>batch:", k, "<===================")
    num_epochs = 100
    for epoch in range(num_epochs):
        # forward
        model = model.to(device)
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        # backPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')
torch.save(model.state_dict(), 'stock_price_regressor_32.pth')
'''
# test
model.load_state_dict(torch.load('stock_price_regressor_32.pth'))
model.eval()  # 将模型设置为评估模式
model.to(device)
res = []
# 对测试数据进行预测
for i in range(0, len(X_test)):
    x_test = []
    y_test = []
    if i + batch_size < len(X_test):
        x_test.append(X_test[i:i + batch_size, :, :])
    else:
        x_test.append(X_test[len(X_test) - batch_size:len(X_test), :, :])
        break
    x_t = np.array(x_test)
    X_test_tensor = torch.from_numpy(x_t).float()
    X_test_tensor = X_test_tensor.to(device)
    X_test_tensor = X_test_tensor.squeeze(0)
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted_prices = test_outputs.squeeze().cpu().numpy()
        res.append(predicted_prices[0])
realdata = y[len(y) - len(res):]
data.set_index('trade_date', inplace=True)
plt.plot(realdata, marker='o')
plt.plot(res, marker='x')
plt.show()

# 绘制价格数据
plt.figure(figsize=(10, 5))  # 设置图表大小
data['close'].plot(kind='line', color='blue')  # 使用pandas的plot方法绘制线图

# 设置图表标题和轴标签
plt.title('Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')

plt.grid(True)
plt.show()
