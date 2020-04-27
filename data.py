import concurrent.futures

import numpy as np
import pandas as pd
from tqdm import tqdm


def apply_parallel(grouped_df, func):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        ret_list = list(
            tqdm(
                executor.map(func, [group for name, group in grouped_df]),
                total=len(grouped_df),
            )
        )
    return pd.concat(ret_list)


def fill_groundtruth(data: pd.DataFrame) -> pd.DataFrame:
    n_splits = int(np.ceil(data.shape[0] / 4))
    column_iloc = data.columns.get_loc("notification_sent")

    data.iloc[::n_splits, column_iloc] = 1
    data.iloc[-1, column_iloc] = 1

    data["tours_today"] = list(range(1, data.shape[0] + 1))
    return data


def calculate_day_fractions(data: pd.Series) -> pd.Series:
    seconds_daily = 24 * 60 * 60
    seconds = data.dt.hour * 60 * 60 + data.dt.minute * 60 + data.dt.second
    fraction_of_a_day = seconds / seconds_daily
    return fraction_of_a_day


def trim_subset(data: pd.DataFrame, fraction: float) -> pd.DataFrame:
    subset = int(data.shape[0] * fraction)
    data = data.iloc[-subset:, :]
    return data


def train_test_time_split(dataset, time_column="date", split_last_days=3):
    timedelta = pd.Timedelta(split_last_days, unit="d")
    split_day = dataset[time_column].max() - timedelta

    test_index = dataset[time_column] >= split_day
    train, test = dataset[~test_index], dataset[test_index]

    return train, test


def load_training_dataset(training_set: str, fraction: float, evaluate: bool):
    dataset = pd.read_csv(
        training_set,
        header=None,
        names=["timestamp", "user_id", "friend_id", "friend_name"],
        parse_dates=["timestamp"],
    )
    dataset = trim_subset(dataset, fraction)

    friends_count = (
        dataset.groupby("user_id")["friend_id"].nunique().rename("friends_count")
    )
    dataset = dataset.merge(friends_count, left_on="user_id", right_index=True)

    dataset["time_fraction"] = calculate_day_fractions(dataset["timestamp"])
    dataset["date"] = dataset["timestamp"].dt.date

    dataset["notification_sent"] = 0
    grouped_dataset = dataset.groupby(["user_id", "date"])
    dataset = apply_parallel(grouped_dataset, fill_groundtruth)

    features = ["time_fraction", "friends_count", "tours_today"]
    target = "notification_sent"

    if evaluate:
        train, test = train_test_time_split(dataset)
        return (train[features], train[target], test[features], test[target])
    else:
        return (dataset[features], dataset[target], None, None)
