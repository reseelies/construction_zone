import unittest
import numpy as np

## Import everything into the namespace so that repr's can evaluate
import numpy as np
from numpy import *
from czone.volume import *
from czone.generator import *

import sys

class czone_TestCase(unittest.TestCase):
    def assertArrayEqual(self, first, second, msg=None) -> None:
        "Fail if the two arrays are unequal by via Numpy's array_equal method."
        self.assertTrue(np.array_equal(first, second), msg=msg)

    def assertReprEqual(self, obj, msg=None,) -> None:
        "Fail if the object re-created by the __repr__ method is not equal to the original."
        with np.printoptions(threshold=sys.maxsize, floatmode='unique'):
            self.assertEqual(obj, eval(repr(obj)), msg=msg)

