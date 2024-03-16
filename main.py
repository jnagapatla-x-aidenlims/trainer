# Import structures from other files
from dataset import Dataset
from helpers import sigmoid, sigmoid_prime, error
from layer import Layer
from network import Network
from export import export

# Print manifest
print("\033c", end="", flush=True)
print("Year 4 Mathematical Exploration 2024", end="\033[K\n\a")
print("Janav Nagapatla and Aiden Lim", end="\033[K\n")
print("All rights reserved", end="\033[K\n")

# Import datasets
print("", end="\033[K\n")
print("Importing datasets", end="\033[K\n")

print("> (1/2) Training Dataset", end="\033[K\n")
training_dataset: Dataset = Dataset("train-images.idx3-ubyte",
                                    "train-labels.idx1-ubyte")

print("> (2/2) Testing Dataset", end="\033[K\n")
testing_dataset: Dataset = Dataset("t10k-images.idx3-ubyte",
                                   "t10k-labels.idx1-ubyte")

print("> Successfully imported 2 out of 2 datasets", end="\033[K\n")

# Initialise a network
print("", end="\033[K\n")
print("Creating network", end="\033[K\n")

network: Network = Network(error,
                           [
                               Layer(784, 30, sigmoid, sigmoid_prime),
                               Layer(30, 10, sigmoid, sigmoid_prime)
                           ])

print("> Successfully initialised network with random weights", end="\033[K\n")

# Evaluate the network
print("", end="\033[K\n")
print("Evaluating initial performance", end="\033[K\n")

network.evaluate(testing_dataset)

# Train the network
print("", end="\033[K\n")
print("Training network", end="\033[K\n")

network.train(training_dataset, testing_dataset, 30, 10, 3.0)

# Reevaluate the network
print("", end="\033[K\n")
print("Evaluating final performance", end="\033[K\n")

network.evaluate(testing_dataset)

# Save the network
print("", end="\033[K\n")
print("Saving model", end="\033[K\n")
filename: str = input("> Name of model (press return to abort): \a")
print("", end="\033[K\033[F")

if not filename:
    exit("> The program has been aborted.\033[K")

export(network, filename)
