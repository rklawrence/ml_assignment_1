import os
import sys
import numpy as np
from sklearn import datasets

# import libraries as needed


def readDataLabels():
    # read in the data and the labels to feed into the ANN
    data = datasets.load_digits()
    X = data.data
    y = data.target

    return X, y


def to_categorical(y):
    """Simply translates the inputs to integer representation of
    the number. This will also act as a way to reference the list
    of output neurons where the index cooresponding to a given digit
    will be the output for that digit.

    Args:
        y (np.ndarray): An array of numeric digits held as a number or a string

    Returns:
        (np.ndarray): All of the categories changed to a single digit.
    """

    # Convert the nominal y values tocategorical
    y = y.astype(int)
    return y


def train_test_split(data, labels, n=0.8):  # TODO
    test_data = (list(), list())
    training_data = (list(), list())
    datapoints = len(data)
    for i in range(datapoints):
        if i > datapoints * n:
            test_data[0].append(data[i])
            test_data[1].append(labels[i])
        else:
            training_data[0].append(data[i])
            training_data[1].append(labels[i])

    return training_data, test_data


def normalize_data(data) -> list:  # TODO
    """Takes in each image and normalizes the value for each
    pixel to be in the range [0, 1] instead of the range [0, 255]
    """
    results = list()
    max_value = 16
    min_value = 0
    for image in data:
        for i in range(len(image)):
            image[i] = (image[i] - min_value) / (max_value - min_value)
        results.append(image)

    return results
