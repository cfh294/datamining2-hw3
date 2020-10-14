#!/usr/bin/env python3
"""
digit-classifier.py

Proof of concept for using a model to predict a handwritten number based on the pixels of an image.
"""
import pathlib
import pandas
import pickle
import logging
import csv
import numpy
from statistics import mean
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

logging.basicConfig(level=logging.DEBUG)


def get_data(directory=pathlib.Path(".", "data"), train_pattern="train.csv"):
    """
    Load data in from csv
    """
    train_path = directory / train_pattern
    return pandas.read_csv(str(train_path))

def get_model(train_data, directory=pathlib.Path(".", "models")):
    """
    Retrieve an SVC object. If there is a model saved locally under models/, then this
    will simply unpack the bytes using pickle and return the model object. If not, then 
    it will create a new model from scratch, using the training data provided from Kaggle. 
    """
    path = directory / "model.bin"
    if path.exists():
        logging.info("Loading model from memory.")
        with open(str(path), "rb") as f:
            model = pickle.load(f)
    else:
        logging.info("No model found in memory, creating one now.")
        model = svm.SVC(gamma="scale")
        scaler = MinMaxScaler(feature_range=(0, 255))
        scores = []
        y = train_data.label.to_numpy()
        x = scaler.fit_transform(
            train_data.drop(columns="label")
        )
        logging.info("Training model...")
        kf = KFold(n_splits=10)
        for train_i, test_i in kf.split(x):
            x_train, y_train = x[train_i], y[train_i]
            x_test, y_test = x[test_i], y[test_i]
            model.fit(x_train, y_train)
            scores.append(model.score(x_test, y_test))
        logging.info("Done.")
        logging.info(f"Accuracy: {mean(scores)}")
        with open(str(path), "wb") as f:
            pickle.dump(model, f)
    return model

def make_predictions(model):
    """
    Use a trained model to make predictions on the given test.csv file from Kaggle.
    """
    in_path = str(pathlib.Path(".", "data", "test.csv"))
    out_path = str(pathlib.Path(".", "output", "output.csv"))
    logging.info("Making predictions...")
    with open(in_path, "r") as in_data_file, open(out_path, "w") as out_file:
        i = 1
        writer = csv.writer(out_file)
        writer.writerow(["ImageId", "Label"])
        reader = csv.reader(in_data_file)
        next(reader)
        for row in reader:
            writer.writerow(
                [
                    i,
                    model.predict(numpy.array([row]))[0]
                ]
            )
            i += 1
    logging.info("Done.")

def main():
    """
    Put it all together, grab data, grab model, make predictions
    """
    make_predictions(
        get_model(
            get_data()
        )
    )


if __name__ == "__main__":
    main()
