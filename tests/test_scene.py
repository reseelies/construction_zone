import unittest
import numpy as np
from czone.volume.volume import Volume
from czone.volume.algebraic import Sphere
from czone.scene.scene import Scene
from functools import reduce
from itertools import repeat

seed = 709123
rng = np.random.default_rng(seed=seed)


class Test_Scene(unittest.TestCase):

    def setUp(self):
        self.rng = rng
        self.N_trials = 32


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

    def test_get_priorities(self):

        def get_objects(N_objects, min_priority=-10, max_priority=10):
            for _ in range(N_objects):
                sphere = Sphere(1, np.zeros((3,1)))
                vol = Volume(alg_objects=[sphere], 
                            priority=int(rng.integers(min_priority, max_priority)),)
                yield vol
        

        for _ in range(self.N_trials):
            ## Get test objects and store in buckets by priority
            objs = list(get_objects(64))
            ref_dict = {}
            for o in objs:
                if o.priority in ref_dict:
                    ref_dict[o.priority].append(id(o))
                else:
                    ref_dict[o.priority] = [id(o)]

            ## Create scene and get priority array
            scene = Scene(objects=objs)
            rel_plevels, offsets = scene._get_priorities()

            orig_priorities = [o.priority for o in objs]
            uniq_priorities = np.unique(orig_priorities)
            N_priorities = len(np.unique(orig_priorities))

            ## Check that priority array has been compressed correctly
            self.assertTrue(np.all(rel_plevels >= 0))
            self.assertTrue(len(offsets) == N_priorities + 1)

            ## Check that all objects are accounted for and counted uniquely
            for rel_p, up in enumerate(uniq_priorities):
                test_ids = [id(scene.objects[i])
                            for i in range(offsets[rel_p], offsets[rel_p+1])]
                self.assertEqual(set(test_ids), set(ref_dict[up]))
                self.assertTrue(len(test_ids) == len(ref_dict[up]))








    # def test_add_object(self):
    #     ## init with two spheres
    #     spheres = [Volume(alg_objects=[Sphere(1, np.zeros((3,1)))])]
    #     scene = Scene(objects=spheres)
    #     scene.add_object(Volume(alg_objects=[Sphere(1, np.zeros((3,1)))]))



class Test_PeriodicScene(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
