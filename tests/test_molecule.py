import numpy as np
from czone_test_fixtures import czone_TestCase
from pymatgen.core.structure import Molecule as pmg_molecule

from czone.molecule.molecule import Molecule
from czone.transform.transform import MatrixTransform

seed = 8907190823
rng = np.random.default_rng(seed=seed)


class Test_Molecule(czone_TestCase):
    def setUp(self):
        self.N_trials = 100

    def test_init(self):
        N = 1024

        ## Check basic initialization
        for _ in range(self.N_trials):
            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))
            mol = Molecule(species, positions)

            self.assertTrue(np.allclose(mol.species, species.ravel()))
            self.assertTrue(np.allclose(mol.atoms, positions))
            self.assertReprEqual(mol)

        ## Check input errors with wrong shaped arrays
        init_f = lambda s, p: Molecule(s, p)
        species = rng.integers(1, 119, (N - 1, 1))
        positions = rng.normal(size=(N, 3))
        self.assertRaises(ValueError, init_f, species, positions)

        species = rng.integers(1, 119, (N, 2))
        positions = rng.normal(size=(N, 3))
        self.assertRaises(ValueError, init_f, species, positions)

        # Numpy should raise an error here, for the reshape of positions
        species = rng.integers(1, 119, (N, 1))
        positions = rng.normal(size=(N, 4))
        self.assertRaises(ValueError, init_f, species, positions)

        # Reshape is valid, but the sizes are now incompatible
        N = 30
        species = rng.integers(1, 119, (N, 1))
        positions = rng.normal(size=(N, 4))
        self.assertRaises(ValueError, init_f, species, positions)

    def test_eq(self):
        N = 256

        ## Check set equality
        for _ in range(self.N_trials):
            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))
            ref_mol = Molecule(species, positions)

            new_order = rng.permutation(np.arange(N))
            test_mol = Molecule(species[new_order], positions[new_order])

            self.assertEqual(ref_mol, test_mol)

    def test_copy_constructors(self):
        N = 512
        for _ in range(self.N_trials // 8):
            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))
            mol = Molecule(species, positions)
            new_mol = Molecule.from_molecule(mol)
            self.assertTrue(np.allclose(mol.species, new_mol.species))
            self.assertTrue(np.allclose(mol.atoms, new_mol.atoms))

            ase_mol = mol.ase_atoms
            self.assertTrue(np.allclose(mol.species, ase_mol.get_atomic_numbers()))
            self.assertTrue(np.allclose(mol.atoms, ase_mol.get_positions()))

            ase_mol = Molecule.from_ase_atoms(ase_mol)
            self.assertTrue(np.allclose(mol.species, ase_mol.species))
            self.assertTrue(np.allclose(mol.atoms, ase_mol.atoms))

            pmg_mol = pmg_molecule(list(species[:, 0]), positions)
            test_mol = Molecule.from_pmg_molecule(pmg_mol)

            ref_species = np.array([s.number for s in pmg_mol.species])
            self.assertTrue(np.allclose(test_mol.species, ref_species))
            self.assertTrue(np.allclose(test_mol.atoms, pmg_mol.cart_coords))

        self.assertRaises(
            TypeError, Molecule.from_ase_atoms, pmg_molecule(list(species[:, 0]), positions)
        )
        self.assertRaises(TypeError, Molecule.from_pmg_molecule, mol.ase_atoms)

    def test_updates(self):
        N = 1024
        for _ in range(self.N_trials):
            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))
            mol = Molecule(species, positions)

            new_species = rng.integers(1, 119, (N, 1))
            new_positions = rng.normal(size=(N, 3))

            mol.update_species(new_species)
            mol.update_positions(new_positions)

            self.assertTrue(np.allclose(mol.species, new_species.ravel()))
            self.assertTrue(np.allclose(mol.atoms, new_positions))

        species = rng.integers(1, 119, (N, 1))
        positions = rng.normal(size=(N, 3))
        mol = Molecule(species, positions)
        f_update_species = lambda s: mol.update_species(s)
        f_update_positions = lambda p: mol.update_positions(p)

        bad_species = rng.integers(1, 119, (N - 1, 1))
        self.assertRaises(ValueError, f_update_species, bad_species)

        bad_positions = rng.normal(size=(N - 1, 3))
        self.assertRaises(ValueError, f_update_positions, bad_positions)

    def test_removes(self):
        N = 1024
        for _ in range(self.N_trials):
            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))
            mol = Molecule(species, positions)

            rem_ind = rng.choice(np.arange(N), 128, replace=False)
            mol.remove_atoms(rem_ind)

            keep_ind = set(np.arange(N)).difference(rem_ind)
            ref_species = np.asarray([species[i, 0] for i in keep_ind])
            ref_pos = np.vstack([positions[i, :] for i in keep_ind])

            self.assertTrue(np.allclose(mol.species, ref_species))
            self.assertTrue(np.allclose(mol.atoms, ref_pos))

        species = rng.integers(1, 119, (N, 1))
        positions = rng.normal(size=(N, 3))
        mol = Molecule(species, positions)

        # Test for bad removal indices
        bad_ind = [0, 1, 2, 3, 1025]
        self.assertRaises(IndexError, mol.remove_atoms, bad_ind)

        bad_ind = [0, 1, 2, 3, -1025]
        self.assertRaises(IndexError, mol.remove_atoms, bad_ind)

        # Test for bad choices of new origins
        bad_ind = [0, 1, 2, 3]
        self.assertRaises(TypeError, mol.remove_atoms, *(bad_ind, 5.2))
        self.assertRaises(IndexError, mol.remove_atoms, *(bad_ind, 1025))
        self.assertRaises(IndexError, mol.remove_atoms, *(bad_ind, -1025))
        self.assertRaises(IndexError, mol.remove_atoms, *(bad_ind, 2))

        # Test for origin tracking removal
        bad_ind = [0, 1, 2, 3]
        mol.set_origin(idx=2)
        self.assertRaises(NotImplementedError, mol.remove_atoms, bad_ind)

    def test_orientation(self):
        N = 4
        for _ in range(self.N_trials):
            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))

            scaled_orientation = rng.normal(
                0,
                1,
                (
                    3,
                    3,
                ),
            )
            orientation, _ = np.linalg.qr(scaled_orientation)

            mol = Molecule(species, positions, orientation=orientation)
            self.assertTrue(np.allclose(mol.orientation, orientation))

        f_molecule = lambda x: Molecule(species, positions, orientation=x)
        self.assertRaises(ValueError, f_molecule, np.eye(4))

        bad_eigenvals = np.eye(3) * 2
        self.assertRaises(ValueError, f_molecule, bad_eigenvals)

        s = np.cbrt(1 / 2.0)
        bad_ortho = np.array([[s, s, 0], [0, s, s], [s, 0, s]])
        self.assertRaises(ValueError, f_molecule, bad_ortho)

    def test_origin(self):
        N = 1024
        for _ in range(self.N_trials):
            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))

            # Default origin at grid origin
            mol_0 = Molecule(species, positions)
            self.assertTrue(np.allclose(mol_0.origin, np.zeros((3, 1))))

            # Check consistency between origin index and manual specification
            origin = rng.choice(1024, 1)[0]
            mol_1 = Molecule(species, positions, origin=origin)
            mol_2 = Molecule(species, positions, origin=positions[origin, :])
            self.assertTrue(np.allclose(mol_1.origin, mol_2.origin))

            # Check tracking of origin
            new_positions = np.copy(positions)
            new_positions[origin, :] = 0.0
            mol_1.update_positions(new_positions)
            self.assertTrue(np.allclose(mol_1.origin, np.zeros((3, 1))))

        f_molecule = lambda x: Molecule(species, positions, origin=x)
        self.assertRaises(IndexError, f_molecule, 1025)
        self.assertRaises(IndexError, f_molecule, -1025)

        f_set_origin = lambda x: mol_0.set_origin(idx=x)
        self.assertRaises(TypeError, f_set_origin, 73.103)

    def test_transform(self):
        N = 1024
        for _ in range(self.N_trials):
            mat = rng.normal(size=(3, 3))
            T = MatrixTransform(mat)

            species = rng.integers(1, 119, (N, 1))
            positions = rng.normal(size=(N, 3))
            mol = Molecule(species, positions)

            test_mol = Molecule.from_molecule(mol, transformation=[T])
            test_mol.set_origin(idx=4)
            mol.set_origin(positions[4, :])
            mol.transform(T)

            self.assertTrue(np.allclose(mol.atoms, (mat @ positions.T).T))
            self.assertTrue(np.allclose(mol.atoms, test_mol.atoms))
            self.assertTrue(np.allclose(mol.origin, test_mol.origin))

        mol.set_origin(idx=4)
        mol.print_warnings = True
        self.assertWarns(UserWarning, mol.transform, T)

        # Since the warnings are only used here, test to make sure warning setting catches errors
        def set_warnings(val):
            mol.print_warnings = val

        self.assertRaises(TypeError, set_warnings, 1.2)
