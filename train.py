import argparse

from joblib import dump
from sklearn import tree
from sklearn.metrics import accuracy_score

from data import load_dataset


def train_and_save(training_set: str, model_file: str, fraction: float, evaluate: bool):
    """Train the model and save it

    Arguments:
        training_set {str} -- URL/path of the dataset
        model_file {str} -- path for the output model file
        fraction {float} -- fraction of the dataset to use
        evaluate {bool} -- should the model be evaluated on held-out set
    """
    train_X, train_y, test_X, test_y = load_dataset(training_set, fraction, evaluate)

    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(train_X, train_y)

    if evaluate:
        y_pred = classifier.predict(test_X)
        accuracy = accuracy_score(test_y, y_pred)

        print(f"Accuracy score: {accuracy}")

    dump(classifier, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train notifications bundling model")
    parser.add_argument("training_set", help="URL/filename of training dataset")
    parser.add_argument(
        "--output-model-file",
        help="Target path for model file",
        default="models/model.joblib",
    )
    parser.add_argument(
        "--subset-fraction",
        help="Fraction of the dataset to use for training (0-1)",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--evaluate",
        help="Calculate evaluation metrics on held-out dataset",
        action="store_true",
    )

    args = parser.parse_args()
    train_and_save(
        args.training_set, args.output_model_file, args.subset_fraction, args.evaluate
    )
