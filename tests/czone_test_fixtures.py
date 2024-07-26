import sys
import unittest

## Import everything into the namespace so that repr's can evaluate
import numpy as np
from numpy import array

from czone.generator.generator import AmorphousGenerator, Generator, NullGenerator
from czone.molecule.molecule import Molecule
from czone.scene.scene import PeriodicScene, Scene
from czone.transform.post import ChemicalSubstitution
from czone.transform.strain import HStrain
from czone.volume.algebraic import Cylinder, Plane, Sphere
from czone.volume.volume import MultiVolume, Volume
from czone.volume.voxel import Voxel
from pymatgen.core import Structure, Lattice


class czone_TestCase(unittest.TestCase):
    def assertArrayEqual(self, first, second, msg=None) -> None:
        "Fail if the two arrays are unequal by via Numpy's array_equal method."
        self.assertTrue(np.array_equal(first, second), msg=msg)

    def assertReprEqual(
        self,
        obj,
        msg=None,
    ) -> None:
        "Fail if the object re-created by the __repr__ method is not equal to the original."
        with np.printoptions(threshold=sys.maxsize, floatmode="unique"):
            self.assertEqual(obj, eval(repr(obj)), msg=msg)
