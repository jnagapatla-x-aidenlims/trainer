# trainer
Training system code for an image classification artificial neural network for Core Mathematics Mathematical Exploration<br>
All rights reserved by Janav Nagapatla and Aiden Lim from 4.10 Samuel (2024) @ Anglo-Chinese School (Independent), Singapore

## main.py
This file contains code that tests the network algorithm prior to training, trains the algorithm, and retests the algorithm.<br>
Upon command of the user, the network can be saved to a file format which can be imported into our implementation code.

## dataset.py
This file contains the paradigm of an image and the paradigm of a dataset.<br>
These are containers for NumPy arrays.<br>
Given a dataset file obtained from the MNIST webpage, image objects will automatically be parsed and created.

## helpers.py
This file contains network helper functions (activation functions and their derivatives) and error functions.

## layer.py
This file contains the paradigm of a fully connected layer.<br>
A layer is a class of NumPy arrays of weights and biases.<br>
Functions of the class include forward and backward passes.

## network.py
This file contains the paradigm of a network.<br>
A network is a collection of layers.<br>
Functions of the class include prediction and training.

## export.py
This file contains network export function.