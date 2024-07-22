from __future__ import annotations
from typing import List, Union

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.symbols import Symbols

from abc import ABC, abstractmethod

from ..volume import BaseVolume, makeRectPrism, Voxel
from czone.transform.transform import Translation
from functools import reduce
from itertools import product

from czone.util.eset import EqualSet, array_set_equal

class BaseScene(ABC):

    def __init__(self):
        self._objects = []

    @property
    def objects(self) -> List[BaseVolume]:
        """List of objects in current scene."""
        return self._objects

    def add_object(self, ob: Union[BaseVolume, List[BaseVolume]]):
        """Add an object to the scene.
        
        Args:
            ob (BaseVolume): object or lists of objects to add to scene.
        """
        if ob is not None:
            ## create a temporary list to handle either single input or iter of inputs
            new_objects = []
            try:
                new_objects.extend(ob)
            except TypeError:
                new_objects.append(ob)

            ## check each new object to see if it is a Volume
            type_check = reduce(lambda x, y: x and y,
                                [isinstance(new_obj, BaseVolume) for new_obj in new_objects])
            if type_check:
                self._objects.extend(new_objects)
            else:
                raise TypeError(f'Object {ob} must inherit from BaseVolume, and is instead {type(ob)}.')

    @property
    def _checks(self):
        """List of logical arrays indicating inclusion of atoms in scene from each object."""
        return self.__checks

    @_checks.setter
    def _checks(self, val):
        self.__checks = val

    #TODO: any way to cache these? is it worth it?
    @property
    def all_atoms(self):
        """Positions of all atoms currently in the scene after evaluating conflict resolution."""
        return np.vstack(
            [ob.atoms[self._checks[i], :] for i, ob in enumerate(self.objects)])

    @property
    def all_species(self):
        """Atomic numbers of all atoms currently in the scene after evaluating conflict resolution."""
        return np.hstack(
            [ob.species[self._checks[i]] for i, ob in enumerate(self.objects)])
    
    def species_from_object(self, idx: int):
        """Grab all the atoms from contributing object at idx.
        
        Returns:
            Numpy array of all positions of atoms contributed by object at idx.
        """
        return self.objects[idx].atoms[self._checks[idx], :]

    def _get_priorities(self):
        """Grab priority levels of all objects in Scene to determine precedence relationship.

        Returns:
            List of relative priority levels and offsets. Relative priority levels
            and offsets are used to determine which objects whill be checked
            for the inclusion of atoms in the scene of the atoms contributed by
            another object.
        
        """
        # get all priority levels active first
        self.objects.sort(key=lambda ob: ob.priority)
        plevels = np.array([x.priority for x in self.objects])

        # get unique levels and create relative priority array
        __, idx = np.unique(plevels, return_index=True)
        rel_plevels = np.zeros(len(self.objects)).astype(int)
        for i in idx[1:]:
            rel_plevels[i:] += 1

        offsets = np.append(idx, len(self.objects))

        return rel_plevels, offsets

    @abstractmethod
    def check_against_object(self, atoms, idx):
        """Check to see if atoms are exterior to object at idx"""
        pass

    @abstractmethod
    def _prepare_for_population(self):
        pass
    
    def populate(self, check_collisions=True):
        """Populate the scene with atoms according to Volumes and priority levels.

        First, every object populates atoms against its own boundaries.
        Then, gather the list of priorities from all the objects.
        For each object, generate a True array of length ob.atoms. 
        For each object in the same priority level or lower, perform interiority 
        check and repeatedly perform logical_and to see if atoms belong in scene.

        - Lower priority numbers supercede objects with high priority numbers.
        - Objects on the same priority level will not supply atoms to the scene in their volume intersections.
        """

        self._prepare_for_population()
        for ob in self.objects:
            ob.populate_atoms()


        ## Sort objects by precedence and get packed list representation
        # offsets is array of length N_priority_levels + 1, 
        # rel_plevels is array of length N_objects, where priorities are >= 0
        rel_plevels, offsets = self._get_priorities()

        self._checks = []

        ## TODO: add some heuristic checking for object collision,
        ## otherwise, with many objects, a lot of unneccesary checks
        for i, ob in enumerate(self.objects):
            check = np.ones(ob.atoms.shape[0]).astype(bool)

            if check_collisions:
                # Grab the final index of the object sharing current priority level
                eidx = offsets[rel_plevels[i] + 1]

                # Iterate over all objects up to priority level and check against their volumes
                for j in range(eidx):
                    if (i != j): # Required, since checking all objects with p_j <= p_i
                        check = np.logical_and(check, 
                                               self.check_against_object(ob.atoms, j))

            self._checks.append(check)


    def populate_no_collisions(self):
        """Populate the scene without checking for object overlap. Use only if known by construction
            that objects have no intersection."""
        self.populate(check_collisions=False)

    def to_file(self, fname, **kwargs):
        """Write atomic scene to an output file, using ASE write utilities.

        If format="prismatic", will default to Debye-Waller factors of 0.1 RMS 
        displacement in squared angstroms, unless dictionary of debye-waller factors
        is otherwise supplied.
        
        Args:
            fname (str): output file name.
            **kwargs: any key word arguments otherwise accepted by ASE write.
        """
        # TODO: refactor and allow for dwf to be specified
        if "format" in kwargs.keys():
            if kwargs["format"] == "prismatic":
                dwf = set(self.all_species)
                dw_default = (0.1**2.0) * 8 * np.pi**2.0
                dwf = {str(Symbols([x])): dw_default for x in dwf}
                ase_write(filename=fname,
                          images=self.ase_atoms,
                          debye_waller_factors=dwf,
                          **kwargs)
        else:
            ase_write(filename=fname, images=self.ase_atoms, **kwargs)


class Scene(BaseScene):
    """Scene classes manage multiple objects interacting in space with cell boundaries.

    Attributes:
        bounds (np.ndarray): 2x3 array defining rectangular bounds of scene.
        objects (List[BaseVolume]): List of all objects currently in scene.
        all_atoms (np.ndarray): Coordinates of all atoms in scene after precedence checks.
        all_species (np.ndarray): Atomic numbers of all atoms in scene after precedence checks.
        ase_atoms (Atoms): Collection of atoms in scene as ASE Atoms object.
    """

    def __init__(self, domain: Voxel, objects=None):
        super().__init__()

        self.domain = domain
        self.add_object(objects)
        self._checks = []

    def __repr__(self) -> str:
        return f'Scene(domain={repr(self.domain)}, objects={repr(self.objects)})'

    def __eq__(self, other: Scene) -> bool:
        if isinstance(other, Scene):
            domain_check = self.domain == other.domain
            object_check = EqualSet(self.objects) == EqualSet(other.objects)
            return domain_check and object_check
        else:
            return False

    @property
    def domain(self) -> Voxel:
        """Current domain of nanoscale scene."""
        return self._domain

    @domain.setter
    def domain(self, domain: Voxel):
        if isinstance(domain, Voxel):
            self._domain = domain
        else:
            raise TypeError

    @property
    def ase_atoms(self):
        """Collection of atoms in scene as ASE Atoms object."""
        cell_dims = self.domain.sbases.T
        celldisp = self.domain.origin
        return Atoms(symbols=self.all_species,
                     positions=self.all_atoms,
                     cell=cell_dims,
                     celldisp=celldisp)

    def check_against_object(self, atoms, idx):
        return np.logical_not(self.objects[idx].checkIfInterior(atoms))
    
    def _prepare_for_population(self):
        pass

class PeriodicScene(BaseScene):
    
    def __init__(self, domain: Voxel, objects=None, pbc=(True, True, True)):
        super().__init__()
        self.domain = domain
        self.pbc = pbc
        self.add_object(objects)
    
    def __repr__(self) -> str:
        return f'PeriodicScene(domain={repr(self.domain)}, objects={repr(self.objects)}, pbc={self.pbc})'

    def __eq__(self, other: PeriodicScene) -> bool:
        # TODO: a more expansive equality check should check on the folded periodic images of domain and pbc are equal
        if isinstance(other, PeriodicScene):
            domain_check = self.domain == other.domain
            pbc_check = self.pbc == other.pbc
            object_check = EqualSet(self.objects) == EqualSet(other.objects)
            return domain_check and object_check and pbc_check
        else:
            return False

    def _get_periodic_indices(self, bbox):
        """Get set of translation vectors, in units of the domain cell, for all 
        relevant periodic images to generate."""

        cell_coords = self.domain.get_voxel_coords(bbox)

        pos_shifts = cell_coords < 0 # Volume needs to be shifted in positive directions
        neg_shifts = cell_coords >= 1 # Volume needs to be shifted in negative directions

        ps = [np.any(pos_shifts[:, i]) for i in range(cell_coords.shape[1])]
        ns = [np.any(neg_shifts[:, i]) for i in range(cell_coords.shape[1])]

        indices = [[0] for _ in range(cell_coords.shape[1])]
        for i, (p, n) in enumerate(zip(ps, ns)):
            if self.pbc[i]:
                if p and n:
                    raise AssertionError('Points extend through periodic domain')
                if p:
                    N_cells = - np.min(np.floor(cell_coords[:,i]))
                    indices[i] = [N_cells]
                    if (not np.all(pos_shifts[:, i])) or (N_cells > 1):
                        indices[i].append(N_cells - 1)
                if n:
                    N_cells = - np.max(np.floor(cell_coords[:,i]))
                    indices[i] = [N_cells]
                    if (not np.all(neg_shifts[:, i])) or (N_cells < -1):
                        indices[i].append(N_cells + 1)

        periodic_indices = set(product(*indices)).difference([(0,0,0)])
        return periodic_indices


    def _get_periodic_images(self):
        """Get periodic images of all objects."""
        self._periodic_images = {}
        for ob in self.objects:
            self._periodic_images[id(ob)] = []

            ## Determine which periodic images need to be generated
            bbox = ob.get_bounding_box()
            periodic_indices = self._get_periodic_indices(bbox)

            for pidx in periodic_indices:
                ## For each image, get a copy of volume translated to its periodic imnage
                pvec = np.array(pidx, dtype=int).reshape((3, -1))
                tvec = (self.domain.sbases @ pvec).reshape((3))
                transformation = [Translation(tvec)]
                new_vol = ob.from_volume(transformation=transformation)
                self._periodic_images[id(ob)].append(new_vol)

    def _get_folded_positions(self, points):
        domain_coords = self.domain.get_voxel_coords(points)

        fold_boundary = np.ones_like(domain_coords, dtype=bool)
        for i, p in enumerate(self.pbc):
            if not p:
                fold_boundary[:, i] = False

        folded_coords = np.mod(domain_coords, 1.0, out=domain_coords, where=fold_boundary)
        return self.domain.get_cartesian_coords(folded_coords)

    @property
    def periodic_images(self):
        return self._periodic_images

    def check_against_object(self, atoms, idx):
        pkey = id(self.objects[idx])
        return np.logical_not(reduce(lambda x,y: np.logical_or(x,y),
                      [po.checkIfInterior(atoms) for po in self.periodic_images[pkey]],
                      self.objects[idx].checkIfInterior(atoms)
                      ))
    
    def _prepare_for_population(self):
        self._get_periodic_images()

    @property
    def all_atoms(self):
        return self._get_folded_positions(super().all_atoms)

    @property
    def ase_atoms(self):
        """Collection of atoms in scene as ASE Atoms object."""
        return Atoms(symbols=self.all_species,
                     positions=self.all_atoms,
                     cell=self.domain.sbases.T,
                     pbc=self.pbc)