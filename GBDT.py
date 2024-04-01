# 导入必要的库
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


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

    # 对获得的Proxies进行MinMax缩放，并转化为torch.Tensor
    for col in columns:
        if col in drop_columns:
            dataset = dataset.drop(labels=col, axis=1)
            continue

        # dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()

    _X = dataset.drop(labels=["close"], axis=1)
    _y = dataset["close"]

    return _X, _y


def train_test_split(X, y, train_ratio=0.75):
    sequence_len = len(X)
    test_len = int(sequence_len * (train_ratio))
    test_X = X[test_len:]
    test_y = y[test_len:]

    train_X = X[:test_len]
    train_y = y[:test_len]

    return train_X, test_X, train_y, test_y


def normalize(X_train, X_test, y_train, y_test):
    y_test = y_test.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)

    return X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std


def inverse_normalize(scaled_tensor, mean_val, std_val):
    # 逆操作：将缩放后的张量还原为原始范围
    original_tensor = scaled_tensor * std_val + mean_val
    return original_tensor


def GBDT_model(crypto_name, input_type):
    # 读取数据（假设你的数据是一个CSV文件）
    path = "data/" + crypto_name + ".csv"

    X, y = get_data(path=path, input_type=input_type)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)

    # 对数据进行缩放
    X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = normalize(
        X_train, X_test, y_train, y_test
    )

    # 初始化GBDT模型
    gbdt_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=1,
        max_depth=17,
        random_state=100,
    )

    # print(X_train)

    # 训练模型
    y_train_flat = y_train.flatten()
    gbdt_model.fit(X_train, y_train_flat)

    # 在测试集上进行预测
    y_pred = gbdt_model.predict(X_test)
    y_test = y_test.flatten()

    # y_pred = inverse_normalize(y_pred, y_mean, y_std)
    # y_test = inverse_normalize(y_test, y_mean, y_std)

    # 评估模型性能
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    test_r2 = r2_score(y_true=y_test, y_pred=y_pred)
    test_mape = np.mean(np.abs((y_test - y_pred) / y_test))

    return {
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mape": test_mape,
    }


if __name__ == "__main__":
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

    # input_types = ["Trading", "TradingPro", "Attention"]

    # out_str = ""
    # for crypto_num in tqdm(range(len(cryptos)), desc=f"GBDT Processing"):
    #     crypto_name = cryptos[crypto_num]
    #     out_str += crypto_name
    #     for input_type in input_types:
    #         res = GBDT_model(crypto_name, input_type)
    #         out_str += f"; {res['test_rmse']}+{res['test_r2']}"
    #         print(crypto_name, input_type, res)

    #     else:
    #         out_str += "\n"

    # with open("results\GBDT.txt", "w") as f:
    #     f.write(out_str)

    name = "GMT"

    res1 = GBDT_model(name, "Trading")
    res2 = GBDT_model(name, "TradingPro")
    res3 = GBDT_model(name, "Attention")

    print(res1)
    print(res2)
    print(res3)
