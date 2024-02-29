from image_tools import Dataset
from connected import ConnectedLayer
from activation import ActivationLayer
from activations import tanh, tanh_prime
from loss import loss, error
from network import Network

training_dataset = Dataset("train-images.idx3-ubyte",
                           "train-labels.idx1-ubyte")
testing_dataset = Dataset("t10k-images.idx3-ubyte",
                          "t10k-labels.idx1-ubyte")

network = Network(loss, error,
                  ConnectedLayer(len(training_dataset.images[0].pixels), 100),
                  ActivationLayer(100, tanh, tanh_prime),
                  ConnectedLayer(100, 50),
                  ActivationLayer(50, tanh, tanh_prime),
                  ConnectedLayer(50, 10),
                  ActivationLayer(10, tanh, tanh_prime),
                  )

# Evaluate the layer

# Train the layer

# Evaluate the layer again