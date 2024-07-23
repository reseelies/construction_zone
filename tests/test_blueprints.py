from czone_test_fixtures import czone_TestCase

import numpy as np

from czone.generator import NullGenerator

from test_generator import get_random_generator, get_random_amorphous_generator
from test_scene import get_random_scene
from test_volume import get_random_volume

from czone.blueprint import Blueprint, Serializer

from pathlib import Path

seed = 9815108923
rng = np.random.default_rng(seed=seed)


class Test_Blueprint(czone_TestCase):
    """object -> blueprint -> object"""
    def setUp(self):
        self.N_trials = 32

    def test_generator(self):
        n_G = NullGenerator()
        n_blueprint = Blueprint(n_G)
        self.assertEqual(n_G, n_blueprint.to_object())
        self.assertReprEqual(n_G)

        for _ in range(self.N_trials):
            a_G = get_random_amorphous_generator(rng=rng)
            a_blueprint = Blueprint(a_G)
            self.assertReprEqual(a_G)
            self.assertEqual(a_G, a_blueprint.to_object())

            G = get_random_generator(rng=rng)
            blueprint = Blueprint(G)
            self.assertReprEqual(G)
            self.assertEqual(G, blueprint.to_object())

    def test_volume(self):
        for _ in range(self.N_trials):
            V = get_random_volume()
            V._alg_objects = []
            blueprint = Blueprint(V)
            self.assertEqual(V, blueprint.to_object())

    def test_scene(self):
        for _ in range(self.N_trials):
            S = get_random_scene()
            blueprint = Blueprint(S)
            self.assertEqual(S, blueprint.to_object())

class Test_Serializer(czone_TestCase):
    """blueprint -> serialized form -> blueprint"""
    def setUp(self):
        self.N_trials = 32
        self.formats = ['h5', 'json']

    def test_generator(self):
        for _ in range(self.N_trials):
            G = get_random_generator()
            blueprint = Blueprint(G)
            for f in self.formats:
                test_path = 'generator_test_file' + '.' + f
                Serializer.write(Path(test_path), blueprint)

                test_G = Serializer.read(Path(test_path)).to_object()
                self.assertEqual(G, test_G)

    def test_volume(self):
        for _ in range(self.N_trials):
            V = get_random_volume()
            blueprint = Blueprint(V)
            for f in self.formats:
                test_path = 'volume_test_file' + '.' + f
                Serializer.write(Path(test_path), blueprint)

                test_V = Serializer.read(Path(test_path)).to_object()
                self.assertEqual(V, test_V)

    def test_scene(self):
        for _ in range(self.N_trials):
            S = get_random_scene()
            blueprint = Blueprint(S)
            for f in self.formats:
                test_path = 'scene_test_file' + '.' + f
                Serializer.write(Path(test_path), blueprint)

                test_S = Serializer.read(Path(test_path)).to_object()
                self.assertEqual(S, test_S)
