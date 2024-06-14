"""
Short module for arbitrary post-generation, pre-volume transformations

Useful for chemical modifciations, statistical defects, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

class BasePostTransform(ABC):
    """Base class for post-generation pre-volume transformations. 
    

    """

    def __init__(self):
        self.origin = np.array([0, 0, 0])

    @abstractmethod
    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):
        """Apply function to a collection of points and species

        Args:
            points (np.ndarray): Nx3 array of points in space
            species (np.ndarray): Nx1 array of corresponding species

        Returns:
            (np.ndarray, np.ndarray): Transformed arrays

        """
        pass

class ChemicalSubstitution(BasePostTransform):

    def __init__(self, mapping : dict, frac, rng=np.random.default_rng()):
        self.target = mapping.keys()
        self.substitute = mapping.values()
        self.frac = frac
        self.rng = rng

    def _replace_species(self, species):

        out_species = np.copy(species)

        for t, s in zip(self.target, self.substitute):
            t_filter = species == t
            t_probs = self.rng.uniform(0,1,size=species.shape)

            out_species[(t_filter) & (t_probs <= self.frac)] = s

        return out_species

    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):
        return points, self._replace_species(species, **kwargs)
        
    @property
    def rng(self):
        """Random number generator associated with Generator"""
        return self._rng
    
    @rng.setter
    def rng(self, new_rng : np.random.BitGenerator):
        if not isinstance(new_rng, np.random.Generator):
            raise TypeError("Must supply a valid Numpy Generator")
        
        self._rng = new_rng



class ArbitraryPostTransform(BasePostTransform):
    def __init__(self, fun):
        self.fun = fun

    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):
        return self.fun(points, species)

class PostSequence(BasePostTransform):
    """Apply sequence of transforms
    
    """

    def __init__(self, transforms: List[BasePostTransform]):
        self._transforms = []
        if not(transforms is None):
            self.add_transform(transforms)

    def add_transform(self, transform: BasePostTransform):
        """Add transform to Multitransform.
        
        Args:
            transform (Basetransform): transform object to add to Multitransform.
        """
        if hasattr(transform, '__iter__'):
            for v in transform:
                assert (isinstance(
                    v, BasePostTransform)), "transforms must be transform objects"
            self._transforms.extend(transform)
        else:
            assert (isinstance(transform,
                               BasePostTransform)), "transforms must be transform objects"
            self._transforms.append(transform)

    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):

        for t in self._transforms:
            points, species = t.apply_function(points, species, **kwargs)

        return points, species

        


