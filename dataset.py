# Import Python libraries
import numpy as np
from typing import BinaryIO


class Image:
    """
    Stores an image and its label from the MNIST datasets from: http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self,
                 label: int,
                 pixels: np.ndarray) -> None:
        """
        Stores the input list into the object
        """

        self.label: int = label
        self.pixels: np.ndarray = pixels

    def __str__(self) -> str:
        """
        Returns a textual representation of the image
        """

        plot: str = ""

        for i, pixel in zip(range(1, 785), self.pixels):
            plot += ("█" if pixel else " ") + ("" if i % 28 else "\n")

        return plot

    def __int__(self) -> int:
        """
        Returns a numeric representation of the image
        """

        return self.label


class Dataset:
    """
    Stores a representation of the MNIST datasets as a list of Image objects
    """

    def __init__(self,
                 images_file: str,
                 labels_file: str) -> None:
        """
        Converts an MNIST dataset file from http://yann.lecun.com/exdb/mnist/ to a dataset structure
        An adaptation of Joseph Redmon's MNIST-to-CSV code from https://pjreddie.com/projects/mnist-in-csv/
        """

        images: BinaryIO = open(images_file, "rb")
        labels: BinaryIO = open(labels_file, "rb")

        images.read(4)

        self.size: int = int.from_bytes(images.read(4))

        images.read(8)
        labels.read(8)

        self.images: list[Image] = [Image(ord(labels.read(1)), np.array([ord(images.read(1)) / 255
                                                                         for _ in range(784)]).reshape(784, 1))
                                    for _ in range(self.size)]

        images.close()
        labels.close()

    def __str__(self) -> str:
        """
        Returns a textual representation of the length of the dataset
        """

        return f"MNIST Dataset of {self.size} images"

    def __int__(self) -> int:
        """
        Returns a numeric representation of the length of the dataset
        """

        return self.size


if __name__ == "__main__":
    match input("Training or Testing Dataset: "):
        case "Training":
            dataset: Dataset = Dataset("train-images.idx3-ubyte",
                                       "train-labels.idx1-ubyte")
        case "Testing":
            dataset: Dataset = Dataset("t10k-images.idx3-ubyte",
                                       "t10k-labels.idx1-ubyte")
        case _:
            exit("That is not a suitable dataset.\033[K")

    try:
        index: int = int(input(f"Which image do you want to plot (1-{dataset.size}): "))

        print(dataset.images[index - 1])
        print(dataset.images[index - 1].label)
    except (ValueError, IndexError):
        exit("That is not a suitable image.\033[K")
