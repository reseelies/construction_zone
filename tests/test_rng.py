import unittest
import numpy as np

from czone.util.misc import get_N_splits

"""
These unit tests are not meant to measure code functionality/correctness.
Instead, these are meant to check that any method using RNGs can be
completely reproduced by passing the RNG in as a property/argument.
"""

seed = 9871492
base_rng = np.random.default_rng(seed=seed)


class Test_Functions(unittest.TestCase):
    def setUp(self):
        self.N_trials = 32

    def assertConsistent(self, F, args, seed):        
        # seed rng and get reference result
        rng = np.random.default_rng(seed)
        ref_state = rng.bit_generator.state # cache state to reseed rng
        ref_res = F(*args, rng=rng)

        # reset RNG state and call function again
        rng.bit_generator.state = ref_state
        test_res = F(*args, rng=rng)
        self.assertEqual(ref_res, test_res)

    def test_get_N_splits(self):
        L = 32
        N = 4
        M = 2
        for _ in range(self.N_trials):
            seed = base_rng.integers(0, int(1e6))
            self.assertConsistent(get_N_splits, (N, M, L), seed)


    
class Test_Classes(unittest.TestCase):
    def setUp(self):
        self.N_trials = 32


