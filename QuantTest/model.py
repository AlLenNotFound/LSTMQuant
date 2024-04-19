import os
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
TotalCapital = 1000000
NetWorth = []


class StockPriceRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_p=0.2):
        super(StockPriceRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)  # 使用最后一个时间步的隐藏状态进行预测
        return out


def create_sequences(data, seq_length):
    result = []
    for index in range(len(data) - seq_length):
        result.append(data[index: index + seq_length])
    return result


def strategy(currentPrice, predicPrice, s, TotalCapital, realPrice):
    if currentPrice < predicPrice:  # 预测需要买入
        tradeAmount = 1 / 3 * TotalCapital
        TotalCapital -= tradeAmount  # 用总金额的1/3买入
        s += tradeAmount / realPrice  # 1/3金额买入的股票数
        NetWorth.append(s * realPrice + TotalCapital)  # 净值
    elif currentPrice >= predicPrice:  # 卖出
        tradeAmount = realPrice * s  # 卖出的实际交易金额
        TotalCapital += tradeAmount  # 卖出后的总金额
        s = 0
        NetWorth.append(s * realPrice + TotalCapital)  # 净值
    return s, predicPrice, TotalCapital


ts_code = '601628.SH.csv'
data = pd.read_csv(ts_code)
data = data[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'turnover_rate', 'volume_ratio', 'pe', 'pb', 'ps',
             'total_mv', 'dv_ratio', 'adj_factor', 'open_hfq', 'open_qfq', 'close_hfq', 'close_qfq', 'high_hfq',
             'high_qfq', 'low_hfq',
             'low_qfq', 'pre_close_hfq', 'pre_close_qfq', 'macd_dif', 'macd_dea', 'macd', 'kdj_k', 'kdj_d',
             'kdj_j', 'rsi_6', 'rsi_12', 'rsi_24', 'boll_upper', 'boll_mid', 'boll_lower', 'cci']]
data['DaysOfWeek'] = pd.to_datetime(data['trade_date']).dt.dayofweek
data['trade_date'] = pd.to_datetime(data['trade_date']).dt.dayofyear
data['cycle'] = data['trade_date'] * (2 * np.pi / 365)
data['cycle'] = np.sin(data['cycle'])
data['Open_Close_Diff'] = data['close'] - data['open']
data['High_Low_Diff'] = data['high'] - data['low']
data['ADay_WaveRate'] = data['High_Low_Diff'] / data['close']

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
num_features = 42
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], seq_length, num_features))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=56)
model = StockPriceRegressor(input_size=num_features, hidden_size=32, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
batch_size = 32
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
        x_t.append(X_train[len(X_train) - batch_size:len(X_train), :, :])
        y_t.append(y[len(X_train) - 1:len(X_train) + batch_size])
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
    num_epochs = 500
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
torch.save(model.state_dict(), 'stock_price_regressor_42Features.pth')
'''
# test
model.load_state_dict(torch.load('stock_price_regressor_42Features.pth'))
model.eval()  # 将模型设置为评估模式
model.to(device)
res = []
DataValidLength = len(X_test)
StartNum = 0

x_t = np.array(X_test)
X_test_tensor = torch.from_numpy(x_t).float()
X_test_tensor = X_test_tensor.to(device)
for i in range(StartNum, StartNum + DataValidLength):
    test_outputs = model(X_test_tensor[i].unsqueeze(0))
    predicted_prices = test_outputs.detach().squeeze().cpu().numpy()
    res.append(predicted_prices)

realData = y_test[StartNum:StartNum + DataValidLength]
# y[len(X_train) + StartNum:len(X_train) + StartNum + DataValidLength]
pastPrice = 9999
# 对测试数据进行预测
stock = 0
benefit = 0
for i in range(0, len(res)):
    stock, pastPrice, TotalCapital = strategy(pastPrice, res[i], stock, TotalCapital, realData[i])

NH = ((NetWorth[-1] - 1000000) / 10000) / DataValidLength * 252
mean = np.mean(res)
squared_diffs = [(x - mean) ** 2 for x in res]
variance = np.mean(squared_diffs)
BD = np.sqrt(variance)
SR = (NH - 4) / BD
y_min, y_max = plt.ylim()
# 计算x轴的中心位置
x_center = (NetWorth[0] + NetWorth[-1]) / 2
Drawdown = min(NetWorth) / 1000000
str = "ReturnRate:{:.2f}% SharpRatio:{:.2f}% Vol:{:.2f}% MaxDrawdown:{:.2f}".format(NH, SR, BD, Drawdown)
plt.title(str)
plt.plot(NetWorth, color='b')
plt.suptitle(ts_code)
plt.axhline(y=1000000, color='r', linestyle='-')
path = os.path.join('./ResShow/' + ts_code + ".jpg")
plt.savefig(path)
plt.show()
