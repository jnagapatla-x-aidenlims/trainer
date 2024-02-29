class Image:
    """
    Stores an image and its label from the MNIST dataset
    MNIST datasets: http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, values):
        """
        Stores the input list into the object
        """
        
        self.number = values[0]
        self.pixels = values[1:-1]

    def __str__(self):
        """
        Returns a textual representation of the image
        """

        return f"Image representing {self.number}"

    def __int__(self):
        """
        Returns a numeric representation of the image
        """

        return self.number

    def plot(self):
        """
        Prints a representation of the image to the console
        for verification of a result

        Only prints in black-and-white
        """

        print(self.number)
        for i in range(len(self.pixels)):
            if i % 28:
                if self.pixels[i]:
                    print("█", end="")
                else:
                    print(" ", end="")
            else:
                if self.pixels[i]:
                    print("█")
                else:
                    print(" ")
  
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

        f = open(images_file, "rb")
        l = open(labels_file, "rb")

        f.read(4)
        
        self.length = int.from_bytes(f.read(4))
        self.images = []

        f.read(8)
        l.read(8)

        for _ in range(self.length):
            image = [ord(l.read(1))]

            for _ in range(784):
                image.append(ord(f.read(1)))

            self.images.append(Image(image))

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
