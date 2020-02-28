# Network setup
# Creates an untrained neural network, according to user preferences
# As of now is also planned to contain simple methods for training the network: random walks and backpropagation
# (Possibly will instead be contained in separate modules)
# 
# 01-02-2020
# Eleonora Svanberg
# Niklas Kastlander

"""
TODO (- assignments)

Implement:
-propagation
-network I/O
-randomization
-backpropagation - Eleonora

Add customization options / User input

Fix minor errors - Niklas

Change data processing to use numpy classes (layer matrices)
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
#from sklearn import preprocessing
#from sklearn.preprocessing import MinMaxScaler
#from sklearn import metrics
#from sklearn.metrics import confusion_matrix
#import itertools

SIZE = [10,5,5,5,2]
# Temporary variable for testing - contains the sizes of each layer

class Layer:
    """
    Abstract class of a network layer
    Contains basic class structure and the constructur method
    """

    _parent = None
    _child = None
    # Connecting each layer to the previous and the next layer in the network

    _matrix = []
    # Contains all mutable layer information (neuron offset, neuron values, and/or weights)

    def __init__(self, parent, size):
        # Connects new layers to a parent, and creates an empty matrix with a number of rows corresponding to size
        self._parent = parent
        self._matrix = []
        for _ in range(size):
            self._matrix.append([])
        if parent:
            parent.fill_matrix(size)
            parent.set_child(self)

    def set_child(self, child):
        self._child = child
    
    def get_matrix(self):
        return self._matrix

    def randomize(self):
        """
        Not implemented
        """
        raise NotImplementedError
    
    def backpropagate(self, matrix, foo):
        """
        Not implemented

        """
        raise NotImplementedError

class NeuronLayer(Layer):
    """
    A layer containing neurons
    """
    
    def fill_matrix(self, size):
        # Fills the matrix with neurons; formatted [current value, offset]
        # Neuronlayers' matrices use only one column, but it would break the symmetry to other fill_matrix methods to not require the size variable
        for neuron in self._matrix:
            neuron += [0,0]
            # NOTE Might want to make the starting values customizable    

class WeightLayer(Layer):
    """
    A layer containing the connection weights between neuron layers
    """

    def fill_matrix(self, size):
        # Generates columns according to size, and fills the matrix with weights
        for row in self._matrix:
            for _ in range(size):
                row.append([0])
                # NOTE Might want to make the starting values customizable

def construct(size_list):

    """
    Constructs the network from size_list, by instatiating NeuronLayers and Weightlayers
    """

    parent = None
    network = []
    for size in size_list:
        i = NeuronLayer(parent,size)
        j = WeightLayer(i, size)
        network += [i,j]
        parent = j
        # BUG Currently the network is generated with one excess WeightLayer - should not be too difficult to fix
    return network

def main():
    network = construct(SIZE)
    for layer in network:
        print(layer.get_matrix())

main()
