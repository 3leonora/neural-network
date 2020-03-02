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

Change fill_matrix system - unnecessary to include on Neuronlayers, so some asymmetry needs to be introduced (possible even in the constructor)

Change data processing to use numpy classes (layer matrices)
"""

import numpy as np 
# import pandas as pd 
import matplotlib.pyplot as plt
#from sklearn import preprocessing
#from sklearn.preprocessing import MinMaxScaler
#from sklearn import metrics
#from sklearn.metrics import confusion_matrix
import itertools
import random

# Import our own helper modules
from .util import pairwise, chunks


SIZE = [10,5,5,5,2]
# Temporary variable for testing - contains the sizes of each layer


class ActivationFunction:
    '''
    Note: The functions must be possible to apply on np arrays.
    '''
    pass


class Sigmoid(ActivationFunction):

    def __call__(self, v) : return 1 / (1 + np.exp(-v))

    def prim(self, v) :
        w = self(v)
        return w * (1 - w)


class LinearNeuronLayer():
    '''
    A layer of linear sum perceptrons with a parametrizable activation
    functor.
    '''
    
    def __init__(self, inoutdims, activation_functor):
        '''
        dims - [indim, outdim]
        '''
        self.afunc = activation_functor
        n, m = inoutdims # m is size of this layer
        self.weights = np.ones([m, n]) # Matrix of ones
        self.biases = np.zeros([m, 1]) # Vector of zeros

    def forward(self, invec):
        '''
        Calculate the forward vector value from this layer.
        '''
        return self.afunc(self.weights.dot(invec) + self.biases) # Linear forward function...

    def randomize(self):
        '''
        Set biases and weights to random values
        '''
        self.weights = np.random.randn(*self.weights.shape)
        self.biases = np.random.randn(*self.biases.shape)

    def print(self):
        print("biases = %s\nweights = %s" % (self.biases, self.weights))



class ANN(object):
    '''
    Construct the full artificial neural network
    Give the constructor a list of layersizes. The first value is however the number of
    input values. The number of layers will be one less that len of the input vector.
    '''
    def __init__(self, layer_sizes):
        self._layer_sizes = layer_sizes
        afunc = Sigmoid()
        self._layers = list(LinearNeuronLayer(dim, afunc) for dim in pairwise(layer_sizes))

        print("Shapes of layers:")
        for n, layer in enumerate(self._layers):
            print("  Layer %d: weight shape=%s, bias shape=%s" % (n, layer.weights.shape, layer.biases.shape))
        

    def forward(self, invec):

        vecval = invec
        for layer in self._layers:
            vecval = layer.forward(vecval)

        return vecval


    def randomize(self):

        for layer in self._layers:
            layer.randomize()

    def print(self):
        for n, layer in enumerate(self._layers):
            print("Layer %d:" % (n,))
            layer.print()


    def stochastic_gradient_descend(self, traindata, numepochs, batch_size, eta):
        '''
        Use gradient descend using sumsquare cost function (hard-coded for now).
        We deploy a stochastic selection of sub-sets of the given training data.

        traindata:  [(invec, outvec), ...] (Must be a list because mutated)
        epochs       Number of batches to process before return
        batch_size   Number of training data samples to use per batch
        eta          learning rate
        '''
        n = len(traindata)
        for epoch in range(numepochs):
            random.shuffle(traindata)
            for (batchnum, batch) in enumerate(chunks(traindata, batch_size)):
                print("Epoch %d, batch %d" % (epoch, batchnum))                
                self.train_batch(batch, eta)

                
    def train_batch(self, batch, eta):

        step_factor = eta / len(batch)

        # Zero-initialize a structure able to contain the full
        # gradient vector (nabla...) over the network's weights and
        # biases.  It is a list of (biasvector, weightmatrices)
        # mimicking the structure of the network itself

        # Note First element in every tuple is the biases.
        grad = [ (np.zeros(layer.biases.shape), np.zeros(layer.weights.shape))
                 for layer in self._layers]

        # Loop over every in/out tuple in batch and sum up the
        # gradient contributions.
        
        for x, y in batch:
            # This is probably the speed bottleneck. Ie that we
            # handle each training datum individually. Maybe this and
            # the backprop function could be changed to process
            # whole matrices of x and y values.
            grad = [ (g[0] + delta_g[0], g[1] + delta_g[1]) for g, delta_g
                     in zip(grad, self.backprop_sumsq(x,y))]
                     
        # Mutate the parameters of the network (biases and weights)

        for layer, layer_grad in zip(self._layers, grad):
            layer.biases -= step_factor * layer_grad[0]
            layer.weights -= step_factor * layer_grad[1]
            


    def backprop_sumsq(self, x, y):
        '''
        Return the gradient of the sumsquare cost function. Ie a structure
        containing all the partial derivative values of the cost function w.r.t
        the weight and bias parameters at the x point. (?)
        '''

        def cost_deriv(output, y): return (output - y)
            
        #        print ("%s -> %s" % (x, y))
        
        # Input x to the network and save some intermediate
        # calculations:
        #   activations: output from from activation funcs
        #                 = input to next layer
        #   zs:          results after weight mult. and bias adjustment
        
        # Note, the activations vector will contain one more vector than
        # the number of layers. The last activations is the output of
        # from the last layer.
        activations = [x]
        zs = []

        activation = x
        
        for layer in self._layers:
            z = layer.weights.dot(activation) + layer.biases # .dot is matrix mult
            zs.append(z)
            activation = layer.afunc(z)
            activations.append(activation)
        
        # Backward pass

        # Prepare the result gradient vector
        grad = [None] * len(self._layers)
                
        # Last layer needs special calculation of 'delta'

        delta = cost_deriv(activations[-1], y) * self._layers[-1].afunc.prim(zs[-1])
        grad[-1] = (delta,                                      # biases gradients
                    np.dot(delta, activations[-2].transpose())) # weights gradients

        for layer_ix in range(2, len(self._layers) + 1):

            delta = np.dot(self._layers[-layer_ix + 1].weights.transpose(), delta) * \
                self._layers[-layer_ix].afunc.prim(zs[-layer_ix])                     

            grad[-layer_ix] = (delta, np.dot(delta, activations[-layer_ix - 1].transpose()))
                    
        return grad
        

# def construct(size_list):
#     '''
#     Constructs the network from size_list, by instatiating NeuronLayers and Weightlayers.
#     '''

#     parent = None
#     network = []
#     for index, size in enumerate(size_list):
#         layer1 = NeuronLayer(parent,size)
#         if index == len(size_list)-1:
#             network += [layer1]
#         else:
#             layer2 = WeightLayer(layer1, size)
#             network += [layer1,layer2]
#             parent = layer2
#     return network

# def main():
#     network = construct(SIZE)
#     for layer in network:
#         print(layer.get_matrix())

# main()
