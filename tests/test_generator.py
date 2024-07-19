
import unittest

import numpy as np
from czone_test_fixtures import czone_TestCase
from numpy import array

from czone.generator import Generator, NullGenerator
from czone.transform import Inversion, Reflection, Rotation, Translation, rot_vtv
from czone.volume.algebraic import Plane

seed = 72349
rng = np.random.default_rng(seed=seed)

def get_transforms():
    for basis_only in [False, True]:
        origin = rng.uniform(-10, 10, size=(1,3))
        yield Inversion(origin=origin, basis_only=basis_only)

        plane = Plane(rng.normal(size=(3,)), rng.normal(size=(3,)))
        yield Reflection(plane, basis_only=basis_only)

        normal = rng.normal(size=(3,))
        R = rot_vtv([0,0,1], normal)
        yield Rotation(R, origin=origin, basis_only=basis_only)

        yield Translation(shift=rng.uniform(-10,10, size=(1,3)), basis_only=basis_only)


def get_transformed_generators(G):
    yield G.from_generator()
    for t in get_transforms():
        yield G.from_generator(transformation=[t])


class Test_NullGenerator(czone_TestCase):
    def test_init_and_eq(self):
        A = NullGenerator()
        B = NullGenerator()
        self.assertEqual(A, B)
        self.assertReprEqual(A)

        for g in get_transformed_generators(A):
            self.assertEqual(A, g)

    def test_supply_atoms(self):
        A = NullGenerator()
        bbox = rng.normal(size=(8, 3))
        pos, species = A.supply_atoms(bbox)
        self.assertEqual(pos.shape, (0, 3))
        self.assertEqual(species.shape, (0,))
        for g in get_transformed_generators(A):
            t_pos, t_species = g.supply_atoms(bbox)
            self.assertArrayEqual(pos, t_pos)
            self.assertArrayEqual(species, t_species)


class Test_Generator(czone_TestCase):
    def setUp(self):
        self.N_trials = 128

    def test_init(self):
        self.assertTrue(False)

    def test_eq(self):
        self.assertTrue(False)
