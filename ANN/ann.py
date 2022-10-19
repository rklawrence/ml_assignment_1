import os
import sys
from turtle import numinput
import numpy as np
import math
import random

from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score, CrossEntropyLoss, ReLUActivation, SoftmaxActivation

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

# train/test... Optional mode to avoid training incase you want to load saved model and test only.
mode = 'train'


class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function

        self.input_to_hidden_weights = None
        self.hidden_to_output_weights = None

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes both the weights for the input to hidden layer
        which is an [num_inputs x num_hidden] shaped matrix and the
        weights for the hidden to output layer which is a
        [num_hidden x num_output] shaped matrix.
        """
        # Initializing as small numbers between [0, 1]
        self.input_to_hidden_weights = np.random.rand(
            self.num_input_features, self.num_hidden_units
        )

        self.hidden_to_output_weights = np.random.rand(
            self.num_hidden_units, self.num_outputs
        )
        return

    def forward(self, x):      # TODO
        """ Takes in an input, runs it through the model and returns the output layer

        Args:
            x (np.ndarray): The input data, it should be a [1 x num_input_features]
            sized matrix
        """
        hidden_layer = np.matmul(x, self.input_to_hidden_weights)
        # TODO: Apply activation function to hidden_layer

        output_layer = np.matmul(hidden_layer, self.hidden_to_output_weights)
        print(output_layer)
        # TODO: Apply activation function to output_layer
        output = self.output_activation(output_layer)
        return output

    def backward(self):     # TODO
        pass

    def update_params(self):    # TODO
        # Take the optimization step.
        return

    def train(self, dataset, learning_rate=0.01, num_epochs=100):
        for epoch in range(num_epochs):
            pass

    def test(self, test_dataset):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        return accuracy


def main(argv):
    ann = ANN(
        num_input_features=64,
        num_hidden_units=16,
        num_outputs=10,
        hidden_unit_activation=ReLUActivation(),
        output_activation=SoftmaxActivation(),
        loss_function=CrossEntropyLoss()
    )

    # Load dataset
    dataset = readDataLabels()      # dataset[0] = X, dataset[1] = y
    normalized_data = normalize_data(dataset[0])
    categorized_labels = dataset[1]
    dataset = (normalized_data, categorized_labels)

    # Split data into train and test split. call function in data.py
    training_data, test_data = train_test_split(dataset[0], dataset[1])
    # print(len(training_data[0]))
    # print(len(test_data[0]))
    normalize_data(training_data[0])

    test_data = training_data[0][1]
    print(ann.forward(test_data))

    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        pass        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.


if __name__ == "__main__":
    main(sys.argv)
