import argparse
from collections import namedtuple
from typing import List

import pandas as pd
from joblib import load
from tqdm import tqdm

from data import load_dataset

TourInfo = namedtuple("TourInfo", ["friend_id", "friend_name", "timestamp"])


def build_message(tours: List[TourInfo]) -> str:
    """Build a notification message for list of tours.

    Arguments:
        tours {List[TourInfo]} -- List of tours

    Returns:
        str -- Notification message
    """
    first_friend = tours[0].friend_name
    tour_friends = set(tour.friend_name for tour in tours)
    message = "went on a tour."
    if len(tours) == 1:
        message = f"{first_friend} {message}"
    elif len(tours) == 2:
        message = f"{first_friend} and 1 other {message}"
    else:
        message = f"{first_friend} and {len(tour_friends)} others {message}"

    return message


def evaluate(csv_file: str, model_file: str, predictions_file: str):
    """Evaluate the test dataset

    Arguments:
        csv_file {str} -- URL/path of the test dataset
        model_file {str} -- Path of the model
        predictions_file {str} -- Predictions CSV file
    """
    print("Loading dataset")
    dataset = load_dataset(csv_file, test_only=True)

    print("Loading model")
    classifier = load(model_file)

    features = ["time_fraction", "friends_count", "tours_today"]
    users = dataset["user_id"].unique()
    tours_users = {user_id: [] for user_id in users}
    sent_notifications = []

    print("Running predictions")
    dataset["notification_sent"] = classifier.predict(dataset[features])

    print("Generating notifications")
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        tours_users[row["user_id"]].append(
            TourInfo(row["friend_id"], row["friend_name"], row["timestamp"])
        )

        if row["notification_sent"]:
            tours = tours_users[row["user_id"]]
            sent_notifications.append(
                {
                    "notification_sent": tours[-1].timestamp,
                    "timestamp_first_tour": tours[0].timestamp,
                    "tours": len(tours),
                    "receiver_id": row["user_id"],
                    "message": build_message(tours),
                }
            )
            tours_users[row["user_id"]] = []
    predictions = pd.DataFrame.from_dict(sent_notifications)
    print(predictions.tail())
    predictions.to_csv(predictions_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run notifications bundling")
    parser.add_argument("csv_file", help="CSV file containing notifications to bundle")
    parser.add_argument(
        "--model-file", help="Path to trained model", default="models/model.joblib"
    )
    parser.add_argument(
        "--predictions-file",
        help="Custom predictions file path",
        default="predictions.csv",
    )

    args = parser.parse_args()
    evaluate(args.csv_file, args.model_file, args.predictions_file)
