import unittest
import numpy as np
from czone.volume.volume import Volume
from czone.volume.algebraic import Sphere
from czone.molecule import Molecule
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


    def test_populate(self):
        def get_objects(N_objects, min_priority=-10, max_priority=10):
            for _ in range(N_objects):
                N = 32
                species = self.rng.integers(1,119,(N,1))
                positions = self.rng.normal(size=(N,3))
                mol = Molecule(species, positions)
                sphere = Sphere(self.rng.uniform(2, 10), 
                                self.rng.uniform(-10, 10, (3,)))
                vol = Volume(alg_objects=[sphere],
                                generator=mol, 
                                priority=int(rng.integers(min_priority, max_priority)),)
                yield vol
            
        def sort_atom_arrays(atoms, species):
            order = np.argsort(atoms[:,0])
            atoms = atoms[order]
            species = species[order]
            return atoms, species


        def get_atoms_from_scene(s):
            test_atoms, test_species = s.all_atoms, s.all_species
            return sort_atom_arrays(test_atoms, test_species)
        
        def get_atoms_from_objects(objs):
            ref_atoms = np.vstack([o.atoms for o in objs])
            ref_species = np.concatenate([o.species for o in objs])
            return sort_atom_arrays(ref_atoms, ref_species)
        
        def brute_force_collision(objs):
            atoms = []
            species = []
            for i, iobj in enumerate(objs):
                current_atoms = iobj.atoms
                current_species = iobj.species
                check = np.ones(current_atoms.shape[0], dtype=bool)
                for j, jobj in enumerate(objs):
                    if i == j:
                        continue

                    if jobj.priority <= iobj.priority:
                        check = np.logical_and(check,
                                               np.logical_not(jobj.checkIfInterior(current_atoms)))

                atoms.append(current_atoms[check, :])
                species.append(current_species[check])

            ref_atoms = np.vstack(atoms)
            ref_species = np.concatenate(species)
            return sort_atom_arrays(ref_atoms, ref_species)


        for _ in range(self.N_trials):
            objs = list(get_objects(32))
            scene = Scene(objects=objs)
            
            ## Populate scene and sort atoms by position
            scene.populate_no_collisions()
            test_atoms, test_species = get_atoms_from_scene(scene)

            ref_atoms, ref_species = get_atoms_from_objects(objs)

            self.assertTrue(np.array_equal(test_atoms, ref_atoms))
            self.assertTrue(np.array_equal(test_species, ref_species))

            ## Check collision handling
            scene.populate()
            test_atoms, test_species = get_atoms_from_scene(scene)
            ref_atoms, ref_species = brute_force_collision(objs)
            self.assertTrue(np.array_equal(test_atoms, ref_atoms))
            self.assertTrue(np.array_equal(test_species, ref_species))






    # def test_add_object(self):
    #     ## init with two spheres
    #     spheres = [Volume(alg_objects=[Sphere(1, np.zeros((3,1)))])]
    #     scene = Scene(objects=spheres)
    #     scene.add_object(Volume(alg_objects=[Sphere(1, np.zeros((3,1)))]))



class Test_PeriodicScene(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
