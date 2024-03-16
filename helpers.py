# Import Python libraries
import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    """
    Returns logistic sigmoid at value
    """

    return 1.0 / (1.0 + np.exp(-value))


def sigmoid_prime(value: np.ndarray) -> np.ndarray:
    """
    Returns the derivative of logistic sigmoid at value
    """

    return (1.0 / (1.0 + np.exp(-value))) * (1 - (1.0 / (1.0 + np.exp(-value))))


def tanh(value: np.ndarray) -> np.ndarray:
    """
    Returns tanh at value
    """

    return np.tanh(value)


def tanh_prime(value: np.ndarray) -> np.ndarray:
    """
    Returns the derivative of tanh at value
    """

    return 1 - np.power(np.tanh(value), 2)


def error(received: np.ndarray,
          answer: int) -> np.ndarray:
    """
    Returns a list of the difference between expected values and predicted values
    """
    desired = np.zeros((10, 1))
    desired[answer] = 1.0

    return received.reshape(10, 1) - desired


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
