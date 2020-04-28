import concurrent.futures
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def apply_parallel(grouped_df: pd.DataFrame, func: Callable) -> pd.DataFrame:
    """Apply a function over grouped dataframe in parallel.

    Arguments:
        grouped_df {pd.DataFrame} -- Grouped DataFrame
        func {Callable} -- Function to apply

    Returns:
        pd.DataFrame -- Dataframe after applying `func`
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        ret_list = list(
            tqdm(
                executor.map(func, [group for name, group in grouped_df]),
                total=len(grouped_df),
            )
        )
    return pd.concat(ret_list, ignore_index=True)


def fill_groundtruth(data: pd.DataFrame) -> pd.DataFrame:
    """Fill groundtruth to the dataset.

    The assumption is that a notification should be sent
    at most four times a day, and as soon as it's possible.

    Arguments:
        data {pd.DataFrame} -- Group of tours for a given user

    Returns:
        pd.DataFrame -- Group of tours with notifications bundled
    """
    if len(data) <= 4:
        data["notification_sent"] = 1
    else:
        n_splits = int(np.ceil(data.shape[0] / 4))
        column_iloc = data.columns.get_loc("notification_sent")

        data.iloc[::n_splits, column_iloc] = 1
        data.iloc[-1, column_iloc] = 1

    data["tours_today"] = list(range(1, data.shape[0] + 1))
    return data


def calculate_day_fractions(data: pd.Series) -> pd.Series:
    """Convert timestamp to day fraction (in range 0-1)

    Arguments:
        data {pd.Series} -- The timestamp column

    Returns:
        pd.Series -- Converted day fraction column
    """
    seconds_daily = 24 * 60 * 60
    seconds = data.dt.hour * 60 * 60 + data.dt.minute * 60 + data.dt.second
    fraction_of_a_day = seconds / seconds_daily
    return fraction_of_a_day


def trim_subset(data: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Trim the dataset to a defined fraction (keep last `fraction`% of data)

    Arguments:
        data {pd.DataFrame} -- The dataset
        fraction {float} -- Fraction to keep (0-1)

    Returns:
        pd.DataFrame -- Trimmed dataset
    """
    subset = int(data.shape[0] * fraction)
    data = data.iloc[-subset:, :]
    return data


def train_test_time_split(
    dataset: pd.DataFrame, time_column: str = "date", split_last_days: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training/tests sets.

    Arguments:
        dataset {[type]} -- The dataset

    Keyword Arguments:
        time_column {str} -- column with datetime
        split_last_days {int} -- number of days in the test set

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] -- Tuple of train, test sets
    """
    timedelta = pd.Timedelta(split_last_days, unit="d")
    split_day = dataset[time_column].max() - timedelta

    test_index = dataset[time_column] >= split_day
    train, test = dataset[~test_index], dataset[test_index]

    return train, test


def load_dataset(
    filename: str,
    fraction: float = 1.0,
    evaluate: bool = False,
    test_only: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]],
    pd.DataFrame,
]:
    """Load the dataset.

    If `evaluation` is `True`, split the dataset into train/test sets.
    If `test_only` is `True`, keep the dataset for predictions.

    The returning sets are X and y (features and target variable)
    OR
    The whole dataset (if `test_only` is `True`)

    Arguments:
        filename {str} -- URL/path of the dataset

    Keyword Arguments:
        fraction {float} -- The fraction of the dataset to keep
        evaluate {bool} -- Should dataset be split to train/test
        test_only {bool} -- Should the whole dataset be kept for prediction

    Returns:
        [type] -- [description]
    """
    dataset = pd.read_csv(
        filename,
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

    features = ["time_fraction", "friends_count", "tours_today"]
    target = "notification_sent"

    dataset["notification_sent"] = 0
    grouped_dataset = dataset.groupby(["user_id", "date"])
    dataset = apply_parallel(grouped_dataset, fill_groundtruth)

    if test_only:
        return dataset

    if evaluate:
        train, test = train_test_time_split(dataset)
        return (train[features], train[target], test[features], test[target])
    else:
        return (dataset[features], dataset[target], None, None)
