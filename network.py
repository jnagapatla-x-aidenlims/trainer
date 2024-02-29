class Network:
    """
    A representation of a network

    Requires the number of input and output neurones of the layer;
    Auto-generates the weights and biases prior to training
    """

    def __init__(self, loss, error, *args):
        """
        Generates random weights and biases (as a starting point) or the layer
        """

        self.loss = loss
        self.error = error

        self.layers = args
        self.size = len(self.layers)

    def __str__(self):
        """
        Returns a textual representation of the network
        """

        return f"This network contains {self.size} total layers"

    def __int__(self):
        """
        Returns a numeric representation of the number of layers in the network
        """

        return self.size
    
    def predict(self, image):
        """
        Forwards the image through the network of layers and returns the output of the last layer
        """
        
        input = image.pixels
    
        for layer in self.layers:
            input = layer.forward(input)
            
        return input
