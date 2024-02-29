from random import random

class ConnectedLayer:
    """
    A representation of a connected layer

    Requires the number of input and output neurones of the layer;
    Auto-generates the weights and biases prior to training
    """

    def __init__(self, input_size, output_size):
        """
        Generates random weights and biases (as a starting point) or the layer
        """

        self.input_size = input_size
        self.output_size = output_size

        self.weights = [[0] * output_size] * input_size
        self.biases = [0] * output_size

        for i in range(self.input_size):
            for o in range(self.output_size):
                self.weights[i][o] = random() - 0.5

        for o in range(self.output_size):
            self.biases[o] = random() - 0.5

    def __str__(self):
        """
        Returns a textual representation of the layer
        """

        return f"This layer contains {self.input_size} input neurones and {self.output_size} output neurones"

    def __int__(self):
        """
        Returns a numeric representation of the number of weights in the layer
        """

        return self.input_size * self.output_size

    def forward(self, input):
        """
        Conducts forward propagation and returns the output neurones based on:

        Input neurones;
        Weights of layer
        """
        output = [0] * self.output_size

        for i in range(self.input_size):
            for o in range(self.output_size):
                output[o] += input[i] * self.weights[i][o]
            output[o] += self.biases[o]

        return output

    def backward(self, output_error, input, rate):
        """
        Conducts backward propagation and returns the derivatives based on:

        Output error;
        Input;
        Rate of change;
        """

        input_error = [0] * self.input_size

        for i in range(self.input_size):
            for o in range(self.output_size):
                input_error[i] += output_error[o] * self.weights[i][o]
        
        for i in range(self.input_size):
            for o in range(self.output_size()):
                self.weights[i][o] -= output_error[o] * input[i] * rate
        
        for o in range(self.output_size):
            self.biases[o] -= output_error[o] * rate

        return input_error
