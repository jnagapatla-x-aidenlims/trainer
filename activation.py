class ActivationLayer:
    """
    A representation of an activation layer

    Requires the number of neurones, an activation function, and its derivative function
    """

    def __init__(self, size, activation, activation_prime):
        """
        Simply assigns parameters for activations
        """

        self.size = size

        self.activation = activation
        self.activation_prime = activation_prime

    def __str__(self):
        """
        Returns a textual representation of the layer
        """

        return f"This layer contains {self.size} input and output neurones"

    def __int__(self):
        """
        Returns a numeric representation of the size of the layer
        """

        return self.size

    def forward(self, input):
        """
        Conducts forward propagation and returns the output neurones based on:

        Input neurones;
        Activation function
        """
        output = [0] * self.size

        for i in range(self.size):
            output[i] += self.activation(input[i])

        return output

    def backward(self, input, error):
        """
        Conducts backward propagation and returns the derivatives based on:

        Input;
        Output error
        """

        input_error = [0] * self.size

        for i in range(self.size):
            input_error[i] = self.activation_prime(input[i]) * error[i]

        return input_error
    