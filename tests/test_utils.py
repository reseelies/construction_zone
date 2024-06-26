import unittest
import numpy as np

## Import everything into the namespace so that repr's can evaluate
from numpy import array
from czone.volume import *

class czone_TestCase(unittest.TestCase):
    def assertArrayEqual(self, first, second, msg=None) -> None:
        "Fail if the two arrays are unequal by via Numpy's array_equal method."
        self.assertTrue(np.array_equal(first, second), msg=msg)

    def assertReprEqual(self, obj, msg=None,) -> None:
        "Fail if the object re-created by the __repr__ method is not equal to the original."
        self.assertEqual(obj, eval(repr(obj)), msg=msg)

