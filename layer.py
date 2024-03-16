# Import Python libraries
import numpy as np
from typing import Callable


class Layer:
    """
    A representation of a connected layer
    Requires the number of input and output neurones of the layer and its desired activation function + derivative
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Callable[[np.ndarray], np.ndarray],
                 activation_prime: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Generates random weights and biases (as a starting point) or the layer
        """

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.activation: Callable[[np.ndarray], np.ndarray] = activation
        self.activation_prime: Callable[[np.ndarray], np.ndarray] = activation_prime

        self.weights: np.ndarray = np.random.randn(output_size, input_size)
        self.biases: np.ndarray = np.random.randn(output_size, 1)

    def forward(self,
                previous: np.ndarray) -> np.ndarray:
        """
        Conducts forward propagation and returns the output neurones
        """

        return self.activation(np.dot(self.weights, previous) + self.biases)

    def nonactivated(self,
                     previous: np.ndarray) -> np.ndarray:
        """
        Conducts forward propagation without activation and returns the output neurones
        """

        return np.dot(self.weights, previous) + self.biases

    def update(self,
               weight_derivatives: np.ndarray,
               bias_derivatives: np.ndarray,
               rate: float,
               size: int) -> None:
        """
        Updates all weights and biases of the layer
        """

        self.weights = self.weights - rate * weight_derivatives / size
        self.biases = self.biases - rate * bias_derivatives / size


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
