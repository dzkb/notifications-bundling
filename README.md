# Notifications Bundling

This repository contains code for notifications bundling solution, utilizing decision trees as a model predicting how individual notifications should be grouped. 

## Model description

This method does not use optimization techniques - these would require knowing what is the exact time of arrival of every tour information in the future. This solution assumes that as tours (individual notifications) arrive, it's impossible to peek into the future, so the only option is to either:
- wait for more tours to come
- bundle already collected tours and send them to user
Such definition makes it easy to treat this problem as a binary classification problem, in which every tour information is classified as final (the notification is sent along with previous tours) or non-final (we need to wait for more tours).

This approach assumes that we know how notifications should be aggregated and sent, i.e. up to four per day, as soon as possible. Having historical data with this exact behavior, it is possible to train a model predicting which action should be taken (wait or send). The simplest model does that by taking into account:
- what time during the day did the tour info arrive
- how many tour notifications for an user have been already received
- how many friends does an user have

For this set of features, a Decision Tree model is constructed and trained. After that, the model is saved for future use.

The model, without heavy parameter tuning achieves accuracy of around 91% (meaning that for 91% of tours, the model correctly decides whether to wait for more, or bundle and send).

## Usage

### 1. Training (optional if using Docker)

Prerequisites: Python 3.8 OR Docker

Install the requirements:

```bash
$ pip install -r requirements.txt
```

Run the training:

```bash
$ python train.py --evaluate <URL/path of the training dataset>
```

### 2. Evaluation (prediction)

```bash
$ python evaluate.py <URL/path of the test dataset>
```

### Running using Docker

Training is part of image build process:

```bash
$ docker build -t notifications --build-arg dataset=<URL/path of the training dataset>
```

To run evaluation, first run a container:

```bash
$ docker run -it notifications
```

and inside the container, run the evaluation script:

```bash
$ python evaluate.py <URL/path of the test dataset>
```

## Remarks

The model is very simple and is more of a proof of concept, than a ready to use solution. The idea is to not peek into the future when bundling notifications. This constraint is broken in one situation: to count the number of friends for every user in the dataset.

When it comes to the predictions, it is expected that some mistakes occur. It is impossible to predict every tour perfectly and a real life solution would require some sort of fallback, i.e. when we hit fourth notification, send no more no matter what.

The solution could've been faster. The reason for that is because it is necessary to iteratively look through users in order to mark when past notifications should've been sent. Improving this means tweaking custom group-by functions in pandas. The good side of this is that training the tree itself is very quick.

Further improvements include:
- introducing new user features
- fine-tuning model's hyperparameters
- adding tests (the solution is quite simple, yet during a normal development process it is a must)
- rebuilding the solution to work in streaming paradigm (currently the solution works on batches of data)
