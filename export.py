# Import Python libraries
from typing import TextIO

# Import structures from other files
from network import Network


def export(network: Network,
           filename: str) -> None:
    """
    Exports the weights and activations of the network and saves the result to the given filename
    """

    try:
        config: TextIO = open(f"{filename}.networkconfig", "x")
    except FileExistsError:
        exit("The file already exists.\033[K")

    print(f"> Created {config.name}", end="\033[K\n")

    config.write(f"Model: {filename}\n")
    config.write("Program: Year 4 Mathematical Exploration 2024\n")
    config.write("Authors: Aiden Lim and Janav Nagapatla\n")
    print("> Manifest written", end="\033[K\n")

    config.write("--- Begin Network Configuration ---\n")

    for layer in network.layers:
        config.write("> New Layer\n")
        config.write(f"    > Input Neurones: {layer.input_size}\n")
        config.write(f"    > Output Neurones: {layer.output_size}\n")
        config.write(f"    > Activation Function: {layer.activation.__name__}\n")

        config.write(f"    > Weights:\n")
        for o in range(layer.output_size):
            for i in range(layer.input_size):
                config.write(f"        > {layer.weights[o, i]}\n")

        config.write(f"    > Biases:\n")
        for o in range(layer.output_size):
            config.write(f"        > {layer.biases[o, 0]}\n")

        print("> Layer written", end="\033[K\n")

    config.write("--- End Network Configuration ---")

    print(f"> The network configuration has been saved to {config.name}", end="\033[K\n")


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
