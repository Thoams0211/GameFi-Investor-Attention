from sklearn.discriminant_analysis import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score


# 单例模式
class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # self.lr = 0.12044
        # self.booster = "gbtree"
        # self.max_depth = 21
        # self.min_child_weight = 3
        # self.gamma = 0
        # self.reg_alpha = 0.0253

        self.lr = 0.5
        self.booster = "gbtree"
        self.max_depth = 21
        self.min_child_weight = 3
        self.gamma = 0
        self.reg_alpha = 0.0256


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

    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)

    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    return X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std


def inverse_normalize(scaled_tensor, mean_val, std_val):
    # 逆操作：将缩放后的张量还原为原始范围
    original_tensor = scaled_tensor * std_val + mean_val
    return original_tensor


def XGBoost_model(crypto_name, input_type):
    # 读取数据（假设你的数据是一个CSV文件）
    path = "data/" + crypto_name + ".csv"

    X, y = get_data(path=path, input_type=input_type)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.75)

    # 对数据进行缩放
    X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = normalize(
        X_train, X_test, y_train, y_test
    )

    # 创建超参数字典
    config = Config()

    # 创建 XGBRegressor 模型
    xgb_regressor = XGBRegressor(
        learning_rate=config.lr,
        booster=config.booster,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        gamma=config.gamma,
        reg_alpha=config.reg_alpha,
    )

    # 训练模型
    xgb_regressor.fit(
        X_train,
        y_train,
        verbose=100,
    )

    # 进行预测
    y_pred = xgb_regressor.predict(X_test)

    # 评估测试集
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    test_mape = np.mean(np.abs((y_test - y_pred) / y_test))

    train_rmse = np.sqrt(mean_squared_error(y_train, xgb_regressor.predict(X_train)))
    train_r2 = r2_score(y_train, xgb_regressor.predict(X_train))

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
    # for crypto_num in tqdm(range(len(cryptos)), desc=f"XGBoost Processing"):
    #     crypto_name = cryptos[crypto_num]
    #     out_str += crypto_name
    #     for input_type in input_types:
    #         res = XGBoost_model(crypto_name, input_type)
    #         out_str += f"; {res['test_rmse']} + {res['test_mape']}"

    #     else:
    #         out_str += "\n"

    # with open("results\XGBoost.txt", "w") as f:
    #     f.write(out_str)

    crypto = "GMT"
    input_types = ["Trading", "TradingPro", "Attention"]

    for input_type in input_types:
        res = XGBoost_model(crypto, input_type)
        print(res)
