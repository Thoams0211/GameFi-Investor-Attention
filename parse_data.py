import pandas as pd
from torch import from_numpy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def get_data(path, input_type):
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

    _X = from_numpy(_X.values).float()
    _y = from_numpy(_y.values).float()

    return _X, _y


def train_test_NN(X, y, train_ratio, batch_size, device):
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

    # print(
    #     "ori_y_mean: ",
    #     ori_y_mean,
    #     "ori_y_std: ",
    #     ori_y_std,
    #     "test_y shape is: ",
    #     test_y.shape,
    #     "test_X shape is: ",
    #     test_X.shape,
    #     "train_y shape is: ",
    #     train_y.shape,
    #     "train_X shape is: ",
    #     train_X.shape,
    # )

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

    train_X = from_numpy(train_X).float().to(device)
    train_y = from_numpy(train_y).float().to(device)
    test_X = from_numpy(test_X).float().to(device)
    test_y = from_numpy(test_y).float().to(device)

    Train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size)

    return Train_loader, test_X, test_y, ori_y_mean, ori_y_std
