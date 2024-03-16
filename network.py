# Import Python libraries
import numpy as np
from typing import Callable
from random import shuffle

# Import structures from other files
from dataset import Image, Dataset
from layer import Layer


class Network:
    """
    A representation of a network
    Takes in the layers of the network and their parameters
    """

    def __init__(self,
                 error: Callable[[np.ndarray, int], np.ndarray],
                 layers: list[Layer]) -> None:
        """
        Creates a set of layers
        """

        self.error: Callable[[np.ndarray, int], np.ndarray] = error
        self.layers: list[Layer] = layers
        self.size: int = len(self.layers)

    def predict(self,
                image: Image) -> np.ndarray:
        """
        Forwards the image through the network of layers and returns the output of the last layer
        """

        evaluation = image.pixels

        for layer in self.layers:
            evaluation = layer.forward(evaluation)

        return evaluation

    def evaluate(self,
                 dataset: Dataset,
                 silent: bool = False) -> float:
        """
        Evaluates the network based on a given dataset
        """

        correct: int = 0

        for image, i in zip(dataset.images, range(1, dataset.size + 1)):
            evaluation: int = int(np.argmax(self.predict(image)))
            if evaluation == image.label:
                correct += 1

            if not silent:
                print(f"> Image {i} / {dataset.size}", end="\033[K\n")
                print(f"> Success {correct / i:.2%}", end="\033[K\n")
                print(f"> |{"█" * round(i / dataset.size * 50)}{"-" * (50 - round(i / dataset.size * 50))}|",
                      end="\033[K\033[F\033[F")

        if not silent:
            print(f"> Out of {dataset.size}, {correct} images were predicted accurately", end="\033[K\n")
            print(f"> That is a {correct / dataset.size:.2%} success rate", end="\033[K\n")

        return correct / dataset.size

    def train(self,
              training_dataset: Dataset,
              testing_dataset: Dataset,
              epochs: int,
              size: int,
              rate: float) -> None:
        """
        Trains the network and updates weights for a number of epochs
        """

        for epoch in range(1, epochs + 1):
            shuffle(training_dataset.images)

            batches = [training_dataset.images[i:i + size] for i in range(0, training_dataset.size, size)]
            for batch in batches:
                self.batch(batch, rate)

            print(f"> Epoch {epoch} / {epochs}", end="\033[K\n")
            print(f"> Success {self.evaluate(testing_dataset, True):.2%}", end="\033[K\n")
            print(f"> |{"█" * round(epoch / epochs * 50)}{"-" * (50 - round(epoch / epochs * 50))}|",
                  end="\033[K\033[F\033[F")

        print(f"> After {epochs} epochs, success reached {self.evaluate(testing_dataset, True):.2%}", end="\033[K\n")

    def batch(self,
              batch: list[Image],
              rate: float) -> None:
        """
        Trains the network and updates weights for a number of epochs
        """

        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]

        for image in batch:
            weight_microgradient, bias_microgradient = self.backprop(image)
            weight_gradient = [wg + wmg for wg, wmg in zip(weight_gradient, weight_microgradient)]
            bias_gradient = [bg + bmg for bg, bmg in zip(bias_gradient, bias_microgradient)]

        for layer in range(self.size):
            self.layers[layer].update(weight_gradient[layer], bias_gradient[layer], rate, len(batch))

    def backprop(self,
                 image: Image) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Returns a tuple representing the gradient for the cost function for an image
        """

        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]

        activations = [image.pixels]
        preactivations = []

        for layer in self.layers:
            preactivations.append(layer.nonactivated(activations[-1]))
            activations.append(layer.activation(preactivations[-1]))

        delta = (self.error(activations[-1], image.label) *
                 self.layers[-1].activation_prime(preactivations[-1]))
        weight_gradient[-1] = np.dot(delta, activations[-2].T)
        bias_gradient[-1] = delta

        for layer in range(2, self.size + 1):
            delta = (np.dot(self.layers[-layer + 1].weights.T, delta) *
                     self.layers[-layer].activation_prime(preactivations[-layer]))
            weight_gradient[-layer] = np.dot(delta, activations[-layer - 1].T)
            bias_gradient[-layer] = delta

        return weight_gradient, bias_gradient


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
