'''
    In the 1x1 space we have two regions with square radius 0.15 at
    points (0.25, 0.25) and (0.75, 0.75)

    Use all 65536 points as training data.

    The input is the coordinate and the output is a single value
    where > 0.5 means inside one of the squares.
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import random
import ann
import sys

# Minumim steps for the king to move...
# Points on an equal distance
# from a center makes up a square
def chessboard_dist(p1, p2): return np.max(np.abs(p1-p2))

def mkcoord(p):
    '''Make a numpy column vector out of a (x,y) image coord'''
    return np.array(p).reshape(len(p),1) / 256.0

_P1 = np.array([[0.25],[0.25]])
_P2 = np.array([[0.75],[0.75]])
_RAD = 0.15


def oracle(coord):
    '''
    Return 1.0 if inside rectangle. 0.0 if outside
    Does not work vectorized...
    '''
    if chessboard_dist(_P1, coord) < _RAD or \
       chessboard_dist(_P2, coord) < _RAD:
        return 1.0
    
    return 0.0

def genimarray(func):

    im = np.zeros([256,256])
    for x, y in it.product(range(256), repeat=2): # Create all pixel coords
        im[x,y] = func(mkcoord((x,y)))
    return im


def run_experiment(args):

    # Give seed to random to make subsequent runs reproducible
    #
    # NOTE: This doesn't seem to make the random calls deterministic. Possible
    # to do that?
    random.seed(123)
    
    print("Classifier1 %s\n" % (args,))
    
    imarr = genimarray(oracle)
    print(imarr)
    plt.figure("True classification")
    plt.imshow(imarr, vmin=0.0, vmax=1.0)

    # input("Press a key to continue")

    training_data = list( (mkcoord(p), np.array([[oracle(mkcoord(p))]]))
                           for p in it.product(range(256), repeat=2))

    # Create a network
    
    network = ann.ANN([2, 10, 10, 1]) # Input dim=2 Output dim = 1
    network.randomize()

    network.print()
    
    # Train it
    network.stochastic_gradient_descend(training_data, 10, 100, 20)

    newim = genimarray(network.forward)

    plt.figure("Trained network classification")    
    plt.imshow(newim, vmin=0.0, vmax=1.0)

    plt.show()
    

    
    
#   x = list(np.array([x,y

    
    
