import numpy as np
import math


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power((y_gt - y_pred), 2)
        return loss

    def grad(self):
        # Derived by calculating dL/dy_pred
        gradient = -1 * (self.current_gt - self.current_prediction)

        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:     # TODO: Make this work!!!
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt
        loss = 0
        for i in range(len(y_pred)):
            loss += y_gt[i] * math.log(y_pred[i])
        loss *= -1/len(y_pred)
        return loss

    def grad(self):
        y_pred = self.current_prediction
        y_gt = self.current_gt
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        gradient = - (y_gt / y_pred) + (1 - y_gt) / (1 - y_pred)
        return gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    """This activation function returns the index of the output that
    has the highest value.
    """

    def __init__(self):
        self.z = None

    def __call__(self, y) -> np.ndarray:
        """Applies the Softmax activation function to the output
        layer of the ANN.

        Args:
            y (np.ndarray): The output layer of the ANN.

        Returns:
            (np.ndarray):
                The results of the softmax function on the input array.
        """
        exponent_sum = 0
        for number in y:
            exponent_sum += math.exp(number)
        results = list()
        for number in y:
            results.append(math.exp(number)/exponent_sum)
        self.z = np.array(results)
        return self.z

    def __grad__(self):
        # num_outputs = len(self.z)
        max_x = np.amax(self.z)
        shifted_sum = 0
        for i in range(len(self.z)):
            shifted_sum += math.exp(self.z[i] - max_x)
        results = list()
        for i in range(len(self.z)):
            results.append(math.exp(self.z[i] - max_x) / shifted_sum)
        # for i in range(len(self.z))
        #     for j in range(len(self.z))
        # partial_derivatives = np.ndarray(
        #     (num_outputs, num_outputs), dtype=float)
        # for i in range(num_outputs):
        #     for j in range(num_outputs):
        #         if i == j:
        #             partial_derivatives[i, j] = self.z[i] * (1 - self.z[j])
        #         else:
        #             partial_derivatives[i, j] = -self.z[i] * self.z[j]
        # return np.sum(partial_derivatives, axis=0)
        return np.array(results)


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    @staticmethod
    def sigmoid(x: float) -> float:
        """Calculates the sigmoid of a given value

        Args:
            x (float): The input for the sigmoid function

        Returns:
            float: The output of the sigmoid function.
        """
        return math.exp(x) / (1 + math.exp(x))

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Applies the sigmoid activation function to the input array.

        Args:
            y (np.ndarray): The 1xN array that acts as the input for
            the activation function.

        Returns:
            np.ndarray: The 1xN array that has had the activation 
            function applied to it.
        """
        self.y = y
        results = list()
        for number in y:
            self.sigmoid(number)
            results.append(self.sigmoid(number))
        self.z = np.array(results)
        return self.z

    def __grad__(self):
        gradient = np.zeros(len(self.y))
        for i, number in enumerate(self.y):
            gradient[i] = self.sigmoid(number) * (1 - self.sigmoid(number))
        return gradient


class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
