import os
import sys
from turtle import numinput
import numpy as np
import math
import random

from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score, CrossEntropyLoss, ReLUActivation, SoftmaxActivation, SigmoidActivation

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

    def forward(self, x) -> np.ndarray:
        """ Takes in an input, runs it through the model and returns the output layer

        Args:
            x (np.ndarray): The input data, it should be a [1 x num_input_features]
            sized matrix
        Return:
            (np.ndarray):
                The output of the ANN
        """
        self.inputs = x
        hidden_layer = np.matmul(x, self.input_to_hidden_weights)
        hidden_layer = self.hidden_unit_activation(hidden_layer)
        self.hidden_values = hidden_layer
        output_layer = np.matmul(hidden_layer, self.hidden_to_output_weights)
        output = self.output_activation(output_layer)
        self.outputs = output
        # self.loss =
        return output

    def backward(self, learning_rate: int):
        # Update the hidden to output weights
        gradient = self.output_activation.__grad__()
        # print(gradient)
        height, width = self.hidden_to_output_weights.shape
        for i in range(height):
            for j in range(width):
                value = self.hidden_to_output_weights[i, j]
                self.hidden_to_output_weights[i, j] = (
                    value - (learning_rate * gradient[j])
                )
        # Update the input to hidden weights
        gradient = self.hidden_unit_activation.__grad__()
        # for i, value in enumerate(gradient):
        #     gradient[i] += np.sum(self.output_activation.__grad__())
        height, width = self.input_to_hidden_weights.shape
        # print(self.input_to_hidden_weights.shape)
        for i in range(height):
            for j in range(width):
                value = self.input_to_hidden_weights[i, j]
                # print(value)
                self.input_to_hidden_weights[i, j] = (
                    value - (learning_rate *
                             gradient[j])
                )

    def update_params(self):    # TODO
        # Take the optimization step.
        return

    def train(self, dataset, learning_rate=.0001, num_epochs=40):
        for epoch in range(num_epochs):
            average_loss = 0
            for i in range(len(dataset[0])):
                image = dataset[0][i]
                label = dataset[1][i]
                output = self.forward(image)
                y_gt = np.zeros(len(output))
                y_gt[label] = 1
                self.loss = self.loss_function(y_pred=output, y_gt=y_gt)
                average_loss += self.loss
                self.backward(learning_rate)
            average_loss = average_loss / len(dataset[0])
            print(f"Epoch {epoch} Average Loss: {average_loss}")
            print(f"Accuracy: {self.test(dataset)}")
        return

    def test(self, test_dataset):
        correct = 0
        for i in range(len(test_dataset[0])):
            output = self.forward(test_dataset[0][i])
            y_true = np.argmax(output)
            if y_true == test_dataset[1][i]:
                correct += 1
        accuracy = correct / len(test_dataset[0])
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        return accuracy


def main(argv):
    ann = ANN(
        num_input_features=64,
        num_hidden_units=16,
        num_outputs=10,
        hidden_unit_activation=SigmoidActivation(),
        output_activation=SoftmaxActivation(),
        loss_function=CrossEntropyLoss()
    )

    # Load dataset
    dataset = readDataLabels()      # dataset[0] = X, dataset[1] = y
    normalized_data = normalize_data(dataset[0])

    categorized_labels = to_categorical(dataset[1])
    dataset = (normalized_data, categorized_labels)

    # Split data into train and test split. call function in data.py
    training_data, test_data = train_test_split(dataset[0], dataset[1])
    # print(len(training_data[0]))
    # print(len(test_data[0]))
    normalize_data(training_data[0])

    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann.train(dataset=dataset, num_epochs=100)
        print(ann.test(test_dataset=test_data))
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.


if __name__ == "__main__":
    main(sys.argv)
