import unittest

import numpy as np
from czone_test_fixtures import czone_TestCase
from test_transform import get_random_mapping

from czone.generator import Generator, NullGenerator
from czone.transform import (
    Inversion,
    Reflection,
    Rotation,
    Translation,
    rot_vtv,
    HStrain,
    ChemicalSubstitution,
)
from czone.volume.algebraic import Plane, Sphere

from pymatgen.core import Lattice, Structure

seed = 72349
rng = np.random.default_rng(seed=seed)


def get_transforms():
    for basis_only in [False, True]:
        origin = rng.uniform(-10, 10, size=(1, 3))
        yield (
            Inversion(origin=origin, basis_only=basis_only),
            f"Inversion with basis_only={basis_only}",
        )

        plane = Plane(rng.normal(size=(3,)), rng.normal(size=(3,)))
        yield (
            Reflection(plane, basis_only=basis_only),
            f"Reflection with basis_only={basis_only}",
        )

        normal = rng.normal(size=(3,))
        R = rot_vtv([0, 0, 1], normal)
        yield (
            Rotation(R, origin=origin, basis_only=basis_only),
            f"Rotation with basis_only={basis_only}",
        )

        yield (
            Translation(shift=rng.uniform(-10, 10, size=(1, 3)), basis_only=basis_only),
            f"Translation with basis_only={basis_only}",
        )


def get_transformed_generators(G):
    yield G.from_generator(), ' Identity'
    for t, msg in get_transforms():
        yield G.from_generator(transformation=[t]), msg


class Test_NullGenerator(czone_TestCase):
    def test_init_and_eq(self):
        A = NullGenerator()
        B = NullGenerator()
        self.assertEqual(A, B)
        self.assertReprEqual(A)

        for g, msg in get_transformed_generators(A):
            self.assertEqual(A, g, msg=f'Failed with {msg}')

    def test_supply_atoms(self):
        A = NullGenerator()
        bbox = rng.normal(size=(8, 3))
        pos, species = A.supply_atoms(bbox)
        self.assertEqual(pos.shape, (0, 3))
        self.assertEqual(species.shape, (0,))
        for g, msg in get_transformed_generators(A):
            t_pos, t_species = g.supply_atoms(bbox)
            self.assertArrayEqual(pos, t_pos, msg=f'Failed with {msg}')
            self.assertArrayEqual(species, t_species, f'Failed with {msg}')


def get_random_generator():
    if rng.uniform() < 0.5:
        hstrain = HStrain(rng.uniform(size=(3,)))
    else:
        hstrain = None

    if rng.uniform() < 0.5:
        chem_sub = ChemicalSubstitution(
            get_random_mapping(rng), frac=rng.uniform()
        )
    else:
        chem_sub = None

    lattice = Lattice(rng.normal(size=(3, 3)))

    N_species = rng.integers(1, 16)
    species = rng.integers(1, 119, size=(N_species))
    pos = rng.uniform(size=(N_species, 3))

    structure = Structure(lattice, species, pos)

    origin = rng.uniform(-10, 10, size=(1, 3))

    return Generator(
        origin=origin,
        structure=structure,
        strain_field=hstrain,
        post_transform=chem_sub,
    )
class Test_Generator(czone_TestCase):
    def setUp(self):
        self.N_trials = 128

    def test_init(self):
        for _ in range(self.N_trials):
            for G, msg in get_transformed_generators(get_random_generator()):
                self.assertReprEqual(G, msg)

        # # type check on structure
        # self.assertRaises(TypeError, lambda x : Generator())

        # # type check on strain_field
        # self.assertRaises(TypeError, lambda x : Generator(strain_field=Inversion(np.zeros(3))))

        # # type check on post_transform
        # self.assertRaises(TypeError, lambda x : Generator(post_transform=Inversion(np.zeros(3)))        
