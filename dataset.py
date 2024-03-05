import numpy

class Image:
    """
    Stores an image and its label from the MNIST dataset
    MNIST datasets: http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, label, pixels):
        """
        Stores the input list into the object
        """
        
        self.label = label
        self.pixels = pixels

    def __str__(self):
        """
        Returns a textual representation of the image
        """
        
        plot = ""

        for i, pixel in zip(range(len(self.pixels)), self.pixels):
            plot += ("█" if pixel else " ") + ("" if i % 28 else "\n")

        return plot

    def __int__(self):
        """
        Returns a numeric representation of the image
        """

        return self.label

    def __array__(self):
        """
        Returns an iterable representation of the image
        """

        return self.pixels

  
class Dataset:
    """
    Stores a representation of the MNIST datasets;
    Contains a list of Image objects
    """

    def __init__(self, images_file, labels_file):
        """
        Converts an MNIST dataset file to a dataset structure;
        MNIST datasets: http://yann.lecun.com/exdb/mnist/
        
        An adaptation of Joseph Redmon's MNIST-to-CSV code;
        Original code: https://pjreddie.com/projects/mnist-in-csv/
        """

        images = open(images_file, "rb")
        labels = open(labels_file, "rb")

        images.read(4)
        
        self.size = int.from_bytes(images.read(4))

        images.read(8)
        labels.read(8)

        self.images = numpy.array([0] * self.size, Image)

        for i in range(self.size):
            self.images[i] = Image(ord(labels.read(1)), numpy.array([ord(images.read(1)) for _ in range(784)]))

    def __str__(self):
        """
        Returns a textual representation of the length of the dataset
        """

        return f"MNIST Dataset of {self.length} images"

    def __int__(self):
        """
        Returns a numeric representation of the length of the dataset
        """

        return self.length
dataset = Dataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")