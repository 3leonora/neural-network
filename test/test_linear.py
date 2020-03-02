import unittest
from unittest import TestCase

from os.path import dirname, join as joinpath

import sys
import numpy as np

sys.path.append(joinpath(dirname(dirname(__file__)), 'pylib'))

# Import of our modules we want to test

import ann

def colvec(vec):
    return np.array(vec).reshape(len(vec),1)
    

class LinearANNTestCase(TestCase):

    def test_simple_creation(self):

        myAnn = ann.ANN([2, 3, 1])

        invec = colvec([0., 0.])
        outvec = myAnn.forward(invec)
        
        self.assertEqual(outvec.shape[0], 1) # Check right dimension of outvec
        
        y = ann.Sigmoid()(1.5)
        self.assertEqual(outvec[0], y)
        
        

        

