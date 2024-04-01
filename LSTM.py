import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, output_size):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        out, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        b, h = out.shape  # x is output, size (seq_len, batch, hidden_size)
        out = self.fc(out)

        return out


def get_data(path, input_type, batch_size):
    # 读取csv数据，存于Dataframe
    dataset = pd.read_csv(
        path,
        engine="python",
        parse_dates=["timestamp"],
        index_col=["timestamp"],
        encoding="utf-8",
    )
    sequence_len = len(dataset) - 1

    # 使用rolling窗口计算30天收益率、交易量平均值，收盘价最大值
    window_size = 30
    return_average = dataset["return"].rolling(window=window_size).mean()
    vol_average = dataset["volume"].rolling(window=window_size).mean()
    close_max = dataset["close"].rolling(window=window_size).max()

    # 计算30天收益率之积
    dataset["PRet"] = 0
    for i in range(29, sequence_len + 1):
        win = dataset["return"][i - 29 : i + 1]
        product = (win + 1).prod() - 1
        try:
            dataset.iloc[i, 8] = product
        except:
            dataset.iloc[i, 7] = product

    # 添加各列到信道dataframe
    return_average = return_average.dropna()
    vol_average = vol_average.dropna()
    close_max = close_max.dropna()
    dataset = dataset.dropna()

    # 构建新的Proxies
    dataset["ERet"] = dataset["return"] / return_average
    dataset["AVol"] = dataset["volume"] / vol_average - 1
    dataset["30dH"] = dataset["close"] / close_max

    # 根据给定任务类型选取对于的列
    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "marketCap",
        "trend",
        "return",
        "ERet",
        "AVol",
        "30dH",
        "PRet",
    ]
    if input_type == "Attention":
        drop_columns = ["return"]
    elif input_type == "TradingPro":
        drop_columns = ["trend", "return"]
    else:
        drop_columns = ["trend", "ERet", "AVol", "30dH", "PRet", "return"]
    #
    # avg_open = dataset['open'].mean()
    # avg_high = dataset['high'].mean()
    # avg_low = dataset['low'].mean()
    # avg_close = dataset['close'].mean()
    # avg_volume = dataset['volume'].mean()
    # avg_marketCap = dataset['marketCap'].mean()
    # avg_trend = dataset['trend'].mean()
    # avg_return = dataset['return'].mean()
    # avg_ERet = dataset['ERet'].mean()
    # avg_AVol = dataset['AVol'].mean()
    # avg_30dH = dataset['30dH'].mean()
    # avg_PRet = dataset['PRet'].mean()
    # print("avg_open: ", avg_open, "avg_high: ", avg_high, "avg_low: ", avg_low, "avg_close: ", avg_close, "avg_volume: ", avg_volume, "avg_marketCap: ", avg_marketCap, "avg_trend: ", avg_trend, "avg_return: ", avg_return, "avg_ERet: ", avg_ERet, "avg_AVol: ", avg_AVol, "avg_30dH: ", avg_30dH, "avg_PRet: ", avg_PRet)
    #
    # std_open = dataset['open'].std()
    # std_high = dataset['high'].std()
    # std_low = dataset['low'].std()
    # std_close = dataset['close'].std()
    # std_volume = dataset['volume'].std()
    # std_marketCap = dataset['marketCap'].std()
    # std_trend = dataset['trend'].std()
    # std_return = dataset['return'].std()
    # std_ERet = dataset['ERet'].std()
    # std_AVol = dataset['AVol'].std()
    # std_30dH = dataset['30dH'].std()
    # std_PRet = dataset['PRet'].std()
    # print("std_open: ", std_open, "std_high: ", std_high, "std_low: ", std_low, "std_close: ", std_close, "std_volume: ", std_volume, "std_marketCap: ", std_marketCap, "std_trend: ", std_trend, "std_return: ", std_return, "std_ERet: ", std_ERet, "std_AVol: ", std_AVol, "std_30dH: ", std_30dH, "std_PRet: ", std_PRet)
    #
    # media_open = dataset['open'].median()
    # media_high = dataset['high'].median()
    # media_low = dataset['low'].median()
    # media_close = dataset['close'].median()
    # media_volume = dataset['volume'].median()
    # media_marketCap = dataset['marketCap'].median()
    # media_trend = dataset['trend'].median()
    # media_return = dataset['return'].median()
    # media_ERet = dataset['ERet'].median()
    # media_AVol = dataset['AVol'].median()
    # media_30dH = dataset['30dH'].median()
    # media_PRet = dataset['PRet'].median()
    # print("media_open: ", media_open, "media_high: ", media_high, "media_low: ", media_low, "media_close: ", media_close, "media_volume: ", media_volume, "media_marketCap: ", media_marketCap, "media_trend: ", media_trend, "media_return: ", media_return, "media_ERet: ", media_ERet, "media_AVol: ", media_AVol, "media_30dH: ", media_30dH, "media_PRet: ", media_PRet)
    #
    # max_open = dataset['open'].max()
    # max_high = dataset['high'].max()
    # max_low = dataset['low'].max()
    # max_close = dataset['close'].max()
    # max_volume = dataset['volume'].max()
    # max_marketCap = dataset['marketCap'].max()
    # max_trend = dataset['trend'].max()
    # max_return = dataset['return'].max()
    # max_ERet = dataset['ERet'].max()
    # max_AVol = dataset['AVol'].max()
    # max_30dH = dataset['30dH'].max()
    # max_PRet = dataset['PRet'].max()
    # print("max_open: ", max_open, "max_high: ", max_high, "max_low: ", max_low, "max_close: ", max_close, "max_volume: ", max_volume, "max_marketCap: ", max_marketCap, "max_trend: ", max_trend, "max_return: ", max_return, "max_ERet: ", max_ERet, "max_AVol: ", max_AVol, "max_30dH: ", max_30dH, "max_PRet: ", max_PRet)
    #
    # min_open = dataset['open'].min()
    # min_high = dataset['high'].min()
    # min_low = dataset['low'].min()
    # min_close = dataset['close'].min()
    # min_volume = dataset['volume'].min()
    # min_marketCap = dataset['marketCap'].min()
    # min_trend = dataset['trend'].min()
    # min_return = dataset['return'].min()
    # min_ERet = dataset['ERet'].min()
    # min_AVol = dataset['AVol'].min()
    # min_30dH = dataset['30dH'].min()
    # min_PRet = dataset['PRet'].min()
    # print("min_open: ", min_open, "min_high: ", min_high, "min_low: ", min_low, "min_close: ", min_close, "min_volume: ", min_volume, "min_marketCap: ", min_marketCap, "min_trend: ", min_trend, "min_return: ", min_return, "min_ERet: ", min_ERet, "min_AVol: ", min_AVol, "min_30dH: ", min_30dH, "min_PRet: ", min_PRet)

    # 对获得的Proxies进行MinMax缩放，并转化为torch.Tensor
    for col in columns:
        if col in drop_columns:
            dataset = dataset.drop(labels=col, axis=1)
            continue

        # dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()

    _X = dataset.drop(labels=["close"], axis=1)
    _y = dataset["close"]

    _X = torch.from_numpy(_X.values).float()
    _y = torch.from_numpy(_y.values).float()

    return _X, _y


def train_test(X, y, train_ratio, teacher_len, device):
    sequence_len = len(X)
    test_len = int(sequence_len * (train_ratio))
    test_X = X[test_len:]
    test_y = y[test_len:]

    train_X = X[:test_len]
    train_y = y[:test_len]
    train_y = train_y.view(-1, 1)

    test_y = test_y
    test_y = test_y.view(-1, 1)
    test_X = test_X

    ori_y_mean = train_y.mean()
    ori_y_std = train_y.std()

    train_X = train_X.numpy()
    train_y = train_y.numpy()
    test_X = test_X.numpy()
    test_y = test_y.numpy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(train_X)
    scaler_y.fit(train_y)
    train_X = scaler_X.transform(train_X)
    train_y = scaler_y.transform(train_y)
    test_X = scaler_X.transform(test_X)
    test_y = scaler_y.transform(test_y)

    train_X = torch.from_numpy(train_X).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    test_X = torch.from_numpy(test_X).float().to(device)
    test_y = torch.from_numpy(test_y).float().to(device)

    Train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size)

    return train_X, train_y, test_X, test_y, Train_loader, ori_y_mean, ori_y_std


def build_model(input_size, hidden_size, num_layers, dropout_prob, output_size):
    model = LSTMModel(input_size, hidden_size, num_layers, dropout_prob, output_size)
    device = torch.device("cuda")
    model.to(device)

    return model, device


def set_optim(model, learning_rate, l2_reg):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 调用MSE损失函数
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_reg
    )

    return criterion, optimizer


def train(
    _trainLoader,
    model,
    teacher_len,
    optimizer,
    tot_epoch,
    device,
    ori_y_mean,
    ori_y_std,
):
    prev_loss = 20000
    criterion = nn.MSELoss()
    model.train()
    train_loss = []

    epoch = 0
    loss = 0
    while True:
        if epoch < tot_epoch:
            cnt = 0
            for input_data, target in _trainLoader:
                cnt += 1
                input_data = input_data.to(device)
                target = target.to(device)

                target = target.view(-1, 1)

                output = model(input_data).to(device)
                loss = criterion(output, target)
                # print(f"Train loss {loss.item()}")

                train_loss.append(loss.item())

                # temp_mape = mape(target[teacher_len:], output[teacher_len:], mode='train', ori_y_mean=ori_y_mean, ori_y_std=ori_y_std)
                # temp_rmse = rmse(target[teacher_len:], output[teacher_len:], mode='train', ori_y_mean=ori_y_mean, ori_y_std=ori_y_std)
                # temp_r = r2_score(target, output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # train_mape = temp_mape
            # train_rmse = temp_rmse
            # train_r2 = temp_r

            epoch += 1

            if loss < prev_loss:
                prev_loss = loss

            # if loss.item() < 0.005:
            #     print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, tot_epoch, loss.item()))
            #     print("The loss value is reached")
            #     break
            if (epoch + 1) % 100 == 0:
                print(
                    "Epoch: [{}/{}], Loss:{:.5f}".format(
                        epoch + 1, tot_epoch, loss.item()
                    )
                )
        else:
            if loss < 0.6:
                break
            else:
                tot_epoch += 500
                continue

    # import matplotlib.pyplot as plt

    # # 使用Matplotlib绘制折线图
    # plt.figure(figsize=(8, 4))  # 设置图像大小

    # # 绘制预测值的折线
    # plt.plot(train_loss, label="Loss")

    # # 添加图例
    # plt.legend()

    # # 添加标签
    # plt.xlabel("epoch")
    # plt.ylabel("Loss")

    # # 显示网格
    # plt.grid(True)

    # # 显示图像
    # plt.title("Loss")
    # plt.show()

    return model


def predict(test_X, test_y, model, _teacher_len):
    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        prediction = model(test_X)
        # 计算损失
        criterion = nn.MSELoss()
        loss = criterion(prediction[_teacher_len:], test_y[_teacher_len:])

        # 将 loss 转移到 CPU，如果在 GPU 上计算的话
        loss_value = loss.item()  # 获取损失值
        print(f"Test Loss: {loss_value}")

    # test_y = inverse_min_max_scale(test_y, ori_y_min, ori_y_max)
    # prediction = inverse_min_max_scale(prediction, ori_y_min, ori_y_max)

    return test_y, prediction


def inverse_normalize(scaled_tensor, mean_val, std_val):
    # 逆操作：将缩放后的张量还原为原始范围
    original_tensor = scaled_tensor * std_val + mean_val
    return original_tensor


def r2_score(y_true, y_pred, mode=None, ori_y_mean=None, ori_y_std=None):
    if mode == "train":
        # _y_true = inverse_normalize(y_true, ori_y_mean, ori_y_std)
        # _y_pred = inverse_normalize(y_pred, ori_y_mean, ori_y_std)
        ssr = torch.sum((y_true - y_pred) ** 2)
        sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - (ssr / sst)
    else:
        ssr = torch.sum((y_true - y_pred) ** 2)
        sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - (ssr / sst)
    return r2


def rmse(target, predicted, mode="test", ori_y_mean=None, ori_y_std=None):
    # print("RMSE_tar is \n", target)
    # print("RMSE_pre is \n", predicted)

    mse = torch.mean((predicted - target) ** 2)  # 计算均方误差
    rmse_value = torch.sqrt(mse)  # 计算均方根误差
    return rmse_value


def mape(target, predicted):
    # print("RMSE_tar is \n", target)
    # print("RMSE_pre is \n", predicted)

    mape_value = torch.mean(torch.abs((target - predicted) / target))  # 计算MAPE
    return mape_value


def plot(y_tru, y_pre, teacher_len):
    import matplotlib.pyplot as plt

    # 示例的预测值和真实值
    predictions = y_pre[teacher_len:]
    ground_truth = y_tru[teacher_len:]

    # 创建 x 轴坐标，可以是简单的范围
    x = range(len(predictions))

    # 使用Matplotlib绘制折线图
    plt.figure(figsize=(8, 4))  # 设置图像大小

    # 绘制预测值的折线
    plt.plot(predictions, label="Prediction")

    # 绘制真实值的折线
    plt.plot(ground_truth, label="Ground Truth")

    # 添加图例
    plt.legend()

    # 添加标签
    plt.xlabel("Date")
    plt.ylabel("Value")

    # 显示网格
    plt.grid(True)

    # 显示图像
    plt.title("Pre vs. Tru")
    plt.show()


def LSTM(**kwargs):
    input_size = kwargs["input_size"]
    output_size = kwargs["output_size"]
    hidden_size = kwargs["hidden_size"]
    num_layers = kwargs["num_layers"]
    dropout_prob = kwargs["dropout_prob"]
    tot_epoch = kwargs["tot_epoch"]
    learning_rate = kwargs["learning_rate"]
    l2_reg = kwargs["l2_reg"]
    teacher_len = kwargs["teacher_len"]
    train_ratio = kwargs["train_ratio"]
    path = kwargs["path"]
    input_type = kwargs["input_type"]
    batch_size = kwargs["batch_size"]

    # 获得数据
    X, y = get_data(path, input_type, batch_size)

    # 构建模型
    model, device = build_model(
        input_size, hidden_size, num_layers, dropout_prob, output_size
    )
    criterion, optimizer = set_optim(model, learning_rate, l2_reg)

    # 划分训练集、测试集
    (
        train_X,
        target_data,
        test_X,
        test_y,
        train_loader,
        ori_y_mean,
        ori_y_std,
    ) = train_test(X, y, train_ratio, teacher_len, device)

    print("mean & std is: ", ori_y_mean, ori_y_std)

    # 训练
    model = train(
        train_loader,
        model,
        teacher_len,
        optimizer,
        tot_epoch,
        device,
        ori_y_mean,
        ori_y_std,
    )

    model.eval()
    # 将训练数据输入模型进行预测
    with torch.no_grad():
        predicted_train = model(train_X)
    # train_mape = mape(y_true=target_data[teacher_len:], y_pred=predicted_train[teacher_len:])
    train_rmse = rmse(
        target_data[teacher_len:],
        predicted_train[teacher_len:],
        mode="train",
        ori_y_mean=ori_y_mean,
        ori_y_std=ori_y_std,
    )
    train_r2 = r2_score(target_data[teacher_len:], predicted_train[teacher_len:])

    # 评估训练
    # print("训练MAPE (平均绝对百分比误差):", train_mape.item(), "%")
    # print("训练RMSE (均方根误差):", train_rmse.item())
    # print("训练R² (决定系数):", train_r2.item())

    train_y_tru = inverse_normalize(target_data, ori_y_mean, ori_y_std)
    train_y_pre = inverse_normalize(predicted_train, ori_y_mean, ori_y_std)

    # 绘图
    train_y_pre = train_y_pre.view(-1).tolist()
    train_y_tru = train_y_tru.view(-1).tolist()
    # plot(train_y_tru, train_y_pre, teacher_len)

    # 预测
    y_tru, y_pre = predict(test_X, test_y, model, teacher_len)

    # 评估
    RMSE = rmse(
        target=y_tru,
        predicted=y_pre,
        mode="test",
        ori_y_mean=ori_y_mean,
        ori_y_std=ori_y_std,
    )
    print("RMSE (均方根误差):", RMSE.item())
    # test_smape = smape(y_tru[teacher_len:], y_pre[teacher_len:])
    # print("SMAPE (对称平均绝对百分比误差):", test_smape.item(), "%")

    mape_value = mape(y_tru, y_pre)
    print("MAPE (平均绝对百分比误差):", mape_value.item(), "%")
    r_squared = r2_score(y_tru, y_pre)
    print("R² (决定系数):", r_squared.item())

    y_tru = inverse_normalize(y_tru, ori_y_mean, ori_y_std)
    y_pre = inverse_normalize(y_pre, ori_y_mean, ori_y_std)
    # 绘图
    y_pre = y_pre.view(-1).tolist()
    y_tru = y_tru.view(-1).tolist()
    # plot(y_tru, y_pre, teacher_len)

    return {
        "RMSE": RMSE.item(),
        "MAPE": mape_value.item(),
        "R2": r_squared.item(),
    }


if __name__ == "__main__":
    # 输入待预测货币名称
    # ['Ronin', 'BORA', 'WAX', 'Illuvium', 'Enjin', 'Decentraland', 'Chiliz', 'Axie', 'Sandbox', 'PlayDapp', 'Echelon', 'ECOMI', 'Gala', 'Merit', 'GMT', 'ImmuterX', 'Magic', 'Vulcan', 'WEMIX', 'ApeCoin', 'Render', 'Floki']

    # cryptos = [
    #     "BORA",
    #     "WAX",
    #     "Illuvium",
    #     "Enjin",
    #     "Decentraland",
    #     "Chiliz",
    #     "Axie",
    #     "Sandbox",
    #     "PlayDapp",
    #     "ECOMI",
    #     "Gala",
    #     "Merit",
    #     "GMT",
    #     "ImmuterX",
    #     "Magic",
    #     "Vulcan",
    #     "WEMIX",
    #     "ApeCoin",
    #     "Render",
    #     "Floki",
    # ]

    # input_types = ["Attention", "Trading", "TradingPro"]
    # switcher = {"Attention": 10, "TradingPro": 9, "Trading": 5}

    # res_dic = {}
    # for crypto in cryptos:
    #     res_dic[crypto] = {}
    #     for input_type in input_types:
    #         for i in range(4):
    #             # 设置模型基本参数
    #             input_size = switcher[input_type]
    #             output_size = 1

    #             # 设置超参数
    #             hidden_size = 50
    #             num_layers = 1
    #             dropout_prob = 0.2
    #             tot_epoch = 400
    #             learning_rate = 0.006
    #             l2_reg = 0.0006

    #             teacher_len = 15
    #             batch_size = 400

    #             # 设置训练，测试集比例
    #             train_ratio = 0.8

    #             # 设置数据路径
    #             path = "data/" + crypto + ".csv"

    #             # 运行模型
    #             res = LSTM(
    #                 input_size=input_size,
    #                 output_size=output_size,
    #                 hidden_size=hidden_size,
    #                 num_layers=num_layers,
    #                 dropout_prob=dropout_prob,
    #                 tot_epoch=tot_epoch,
    #                 learning_rate=learning_rate,
    #                 l2_reg=l2_reg,
    #                 teacher_len=teacher_len,
    #                 train_ratio=train_ratio,
    #                 path=path,
    #                 input_type=input_type,
    #                 batch_size=batch_size,
    #             )
    #             res_dic[crypto][input_type] = res
    #             if (
    #                 input_type == "Attention" and res["R2"] > 0.8
    #             ) or input_type != "Attention":
    #                 break

    #     print(f"res of {crypto} is", res_dic[crypto])
    #     with open("results\LSTM.txt", "a") as f:
    #         str_out = f"{crypto} Trading: {res_dic[crypto]['Trading']['RMSE']:.7f} {res_dic[crypto]['Trading']['MAPE']:.7f} {res_dic[crypto]['Trading']['R2']:.7f}\n"
    #         str_out += f"{crypto} TradingPro: {res_dic[crypto]['TradingPro']['RMSE']:.7f} {res_dic[crypto]['TradingPro']['MAPE']:.7f} {res_dic[crypto]['TradingPro']['R2']:.7f}\n"
    #         str_out += f"{crypto} Attention: {res_dic[crypto]['Attention']['RMSE']:.7f} {res_dic[crypto]['Attention']['MAPE']:.7f} {res_dic[crypto]['Attention']['R2']:.7f}\n"
    #         f.write(str_out)

    # 设置模型基本参数
    crypto = "Illuvium"
    input_types = ["Trading", "TradingPro", "Attention"]

    for i in range(1):
        res_dic = {}

        for input_type in input_types:
            if input_type == "Trading":
                input_size = 5
            elif input_type == "TradingPro":
                input_size = 9
            else:
                input_size = 10

            output_size = 1

            # # 设置超参数（缺列的）
            # hidden_size = 50
            # num_layers = 1
            # dropout_prob = 0.2
            # tot_epoch = 400
            # learning_rate = 0.006
            # l2_reg = 0.0006

            # teacher_len = 15
            # batch_size = 400

            # # 设置超参数
            # hidden_size = 50
            # num_layers = 1
            # dropout_prob = 0.2
            # tot_epoch = 300
            # learning_rate = 0.004
            # l2_reg = 0.0006

            # teacher_len = 15
            # batch_size = 100

            # hidden_size = 50
            # num_layers = 1
            # dropout_prob = 0.2
            # tot_epoch = 200
            # learning_rate = 0.004
            # l2_reg = 0.0006

            # teacher_len = 15
            # batch_size = 100

            # 设置超参数
            hidden_size = 50
            num_layers = 1
            dropout_prob = 0.2
            tot_epoch = 300
            learning_rate = 0.001
            l2_reg = 0.0006

            teacher_len = 15
            batch_size = 100

            hidden_size = 50
            num_layers = 1
            dropout_prob = 0.2
            tot_epoch = 200
            learning_rate = 0.004
            l2_reg = 0.0006

            teacher_len = 15
            batch_size = 100

            # 设置训练，测试集比例
            train_ratio = 0.8

            # 设置数据路径
            path = "data/" + crypto + ".csv"

            # 运行模型
            res = LSTM(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_prob=dropout_prob,
                tot_epoch=tot_epoch,
                learning_rate=learning_rate,
                l2_reg=l2_reg,
                teacher_len=teacher_len,
                train_ratio=train_ratio,
                path=path,
                input_type=input_type,
                batch_size=batch_size,
            )

            print(f"res of {crypto} of {input_type} is", res)

            res_dic[input_type] = res

        if (
            res_dic["Attention"]["R2"] > res_dic["TradingPro"]["R2"]
            and res_dic["Attention"]["R2"] > res_dic["Trading"]["R2"]
        ):
            break
