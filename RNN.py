import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from parse_data import get_data, train_test_NN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class RNNmodel(nn.Module):

    """
    定义模型、网络结构, 包含一个RNN, 一个全连接层, 一个dropout层
    """

    def __init__(
        self, *, input_size, hidden_size, num_layers, output_size, dropout=0.2
    ):
        super(RNNmodel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        # out = self.dropout(out)
        out = self.linear(out)
        out = self.relu(out)
        out = self.dense(out)
        return out


class Config:

    """
    单例模式超参数配置类
    """

    _instancec = None

    def __init__(self) -> None:
        # Attention Hyperparameters
        # self.epoch = 75
        # self.lr = 0.00008
        # self.batch_size = 16
        # self.l2 = 0.00075
        # self.hidden_size = 32
        # self.dropout = 0.2
        # self.num_layers = 2
        # self.train_ratio = 0.8

        # Trading Hyperparameters
        # self.epoch = 200
        # self.lr = 0.00008
        # self.batch_size = 64
        # self.l2 = 0.0008
        # self.hidden_size = 32
        # self.dropout = 0.2
        # self.num_layers = 2
        # self.train_ratio = 0.75

        # self.epoch = 100
        # self.lr = 0.001
        # self.batch_size = 32
        # self.l2 = 0.0008
        # self.hidden_size = 32
        # self.dropout = 0.2
        # self.num_layers = 2
        # self.train_ratio = 0.75

        self.epoch = 350
        self.lr = 0.0005
        self.batch_size = 64
        self.l2 = 0.0008
        self.hidden_size = 32
        self.dropout = 0.2
        self.num_layers = 2
        self.train_ratio = 0.75

    def __new__(cls, *args, **kwargs):
        if cls._instancec is None:
            cls._instancec = super().__new__(cls)
        return cls._instancec


def train(model, train_loader, device):
    """
    训练模型
    """
    # 设置优化器和评价准则
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config().lr, weight_decay=Config().l2
    )

    # 训练
    model.train()
    train_loss = []

    for epoch in tqdm(range(Config().epoch), desc="Training Process"):
        for input_data, target in train_loader:
            # 前向传播
            input_data = input_data.to(device)
            target = target.to(device)
            target = target.view(-1, 1)
            output = model(input_data).to(device)
            loss = criterion(output, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            train_loss.append(loss.item())
    # else:
    #     # 使用Matplotlib绘制折线图
    #     plt.figure(figsize=(8, 4))  # 设置图像大小

    #     # 绘制预测值的折线
    #     plt.plot(train_loss, label="Loss")

    #     # 添加图例
    #     plt.legend()

    #     # 添加标签
    #     plt.xlabel("epoch")
    #     plt.ylabel("Loss")

    #     # 显示网格
    #     plt.grid(True)

    #     # 显示图像
    #     plt.title("Loss")
    #     plt.show()

    return model


def predict(model, test_X, test_y, device):
    model.eval()

    with torch.no_grad():
        pred = model(test_X).to(device)

    # r_square
    ssr = ((pred - test_y) ** 2).sum().item()
    sst = ((test_y - test_y.mean()) ** 2).sum().item()
    r2 = 1 - ssr / sst

    # rmse
    mse = ((pred - test_y) ** 2).mean().item()
    rmse = mse**0.5

    # mape
    mape = torch.mean(torch.abs((pred - test_y) / test_y))

    return r2, rmse, mape


def RNN_main(crypto_name, input_type):
    # 获取数据
    path = f"data\{crypto_name}.csv"
    X, y = get_data(path, input_type)

    # 搭建模型
    model = RNNmodel(
        input_size=X.shape[1],
        hidden_size=Config().hidden_size,
        num_layers=Config().num_layers,
        output_size=1,
        dropout=Config().dropout,
    )
    device = torch.device("cuda")
    model.to(device)

    # 划分训练集和测试集
    train_loader, test_X, test_y, ori_y_mean, ori_y_std = train_test_NN(
        X, y, Config().train_ratio, Config().batch_size, device
    )

    # 训练
    model = train(model, train_loader, device)

    # 预测
    r2, rmse, mape = predict(model, test_X, test_y, device)

    # 设置日志输出格式
    logging.basicConfig(
        filename="logs\RNN.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(
        f"{crypto_name} {input_type} RNN r2: {r2}, rmse: {rmse}, mape: {mape}. Hyperparameters: epoch: {Config().epoch}, lr: {Config().lr}, batch_size: {Config().batch_size}, l2: {Config().l2}, hidden_size: {Config().hidden_size}, dropout: {Config().dropout}, num_layers: {Config().num_layers}, train_ratio: {Config().train_ratio}"
    )

    print(f"{crypto_name} {input_type} RNN r2: {r2}, rmse: {rmse}, mape: {mape}")

    return {
        "r2": r2,
        "rmse": rmse,
        "mape": mape,
    }


if __name__ == "__main__":
    name = "Illuvium"
    for i in range(1):
        res_dic = {}
        for input_type in ["Trading", "TradingPro", "Attention"]:
            res = RNN_main(name, input_type)
            res_dic[input_type] = res

        if (
            res_dic["Attention"]["rmse"] < res_dic["TradingPro"]["rmse"]
            and res_dic["Attention"]["rmse"] < res_dic["Trading"]["rmse"]
        ):
            break
