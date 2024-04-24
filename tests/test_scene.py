import unittest
import numpy as np
from czone.volume.volume import Volume
from czone.volume.algebraic import Sphere
from czone.scene.scene import Scene
from functools import reduce

seed = 709123
rng = np.random.default_rng(seed=seed)


class Test_Scene(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    

    def test_init(self):

        ## init with one sphere
        sphere = Sphere(1, np.zeros((3,1)))
        self.assertRaises(TypeError, lambda : Scene(objects=sphere))

        ## init with two spheres
        spheres = [Volume(alg_objects=[Sphere(1, np.zeros((3,1)))]),
                Sphere(5, np.zeros((3,1)))]
        self.assertRaises(TypeError, lambda : Scene(objects=spheres))

        spheres = [Volume(alg_objects=[Sphere(1, np.zeros((3,1)))]),
                    Volume(alg_objects=[Sphere(5, np.zeros((3,1)))])]
        scene = Scene(objects=spheres)

        ref_ids = [id(x) for x in spheres]
        test_ids = [id(x) for x in scene.objects]
        self.assertEqual(set(ref_ids), set(test_ids))

    # def test_get_priorities(self):

    #     objs  



    # def test_add_object(self):
    #     ## init with two spheres
    #     spheres = [Volume(alg_objects=[Sphere(1, np.zeros((3,1)))])]
    #     scene = Scene(objects=spheres)
    #     scene.add_object(Volume(alg_objects=[Sphere(1, np.zeros((3,1)))]))



class Test_PeriodicScene(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
