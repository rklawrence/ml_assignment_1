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
        loss = 0
        for i in range(len(y_pred)):
            loss += y_gt[i] * math.log(y_pred[i])
        loss *= -1
        return loss

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        gradient = None
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
        return np.array(results)

    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        pass


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        pass

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Applies the sigmoid activation function to the input array.

        Args:
            y (np.ndarray): The 1xN array that acts as the input for
            the activation function.

        Returns:
            np.ndarray: The 1xN array that has had the activation 
            function applied to it.
        """
        results = list()
        for number in y:
            results.append(math.exp(number) / (1 + math.exp(number)))
        return np.array(results)

    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        pass


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
