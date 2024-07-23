from dataclasses import dataclass, field

from abc import ABC, abstractmethod
from collections.abc import Mapping

from czone.util.eset import EqualSet
from czone.volume import BaseVolume, Volume, MultiVolume, Voxel, Plane, Sphere, Cylinder
from czone.scene import BaseScene, Scene, PeriodicScene
from czone.generator import BaseGenerator, AmorphousGenerator, Generator, NullGenerator
from czone.molecule import Molecule
from czone.transform import BaseStrain, BasePostTransform

from pymatgen.core import Structure, Lattice
import numpy as np

@dataclass(kw_only=True)
class BaseNode(ABC, Mapping):
    # metadata: dict = field(default_factory=dict)
    # name: str = ''
    children: tuple = tuple()

    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        pass

    @property
    @abstractmethod
    def base_type(self):
        pass

    @property
    @abstractmethod
    def class_type(self):
        pass

    @abstractmethod
    def add_node(self, new_node):
        pass

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __len__(self):
        return len(self.__dict__)
    
    def __iter__(self):
        yield from self.__dict__
    
class BaseGeneratorNode(BaseNode):

    @property
    def base_type(self):
        return BaseGenerator
    
    @property
    def class_type(self):
        return BaseGenerator

    @property
    def is_leaf(self):
        return True
    
    def add_node(self):
        pass

class NullGeneratorNode(BaseGeneratorNode):
    @property
    def class_type(self):
        return NullGenerator

@dataclass
class AmorphousGeneratorNode(BaseGeneratorNode):
    origin: np.ndarray
    min_dist: float
    density: float
    species: int

    @property
    def class_type(self):
        return AmorphousGenerator
    
@dataclass
class GeneratorNode(BaseGeneratorNode):
    origin: np.ndarray
    lattice_matrix: np.ndarray
    basis_species: tuple[int]
    basis_coords: np.ndarray
    strain_field: BaseStrain | None = None
    post_transform: BasePostTransform | None = None

    @property
    def class_type(self):
        return Generator


class Blueprint():
    """
    Represents (Periodic)Scenes, (Multi)Volumes, and Generators 
    as nested mappings.
    """
    def __init__(self, obj):
        self.get_mapping(obj)

    def __repr__(self, other):
        return f"Blueprint({repr(self.to_object())})"

    def __eq__(self, other):
        if isinstance(other, Blueprint):
            other_obj = other.to_object()
        else:
            other_obj = other

        return self.to_object() == other_obj

    @property
    def mapping(self):
        return self._mapping
    
    @mapping.setter
    def mapping(self, node):
        if isinstance(node, BaseNode):
            self._mapping = node
        else:
            raise TypeError

    def map_generator(self, G: BaseGenerator):
        match G:
            case NullGenerator():
                return NullGeneratorNode()
            case AmorphousGenerator():
                return AmorphousGeneratorNode(G.origin, G.min_dist, G.density, G.species)
            case Generator():
                params = {'origin':G.origin,
                          'lattice_matrix':G.lattice.matrix,
                          'basis_species':G.species,
                          'basis_coords':G.coords,
                          'strain_field':G.strain_field,
                          'post_transform':G.post_transform}
                return GeneratorNode(**params)
            case _:
                raise NotImplementedError
            
    def map_volume(self, V):
        # Should be recursive, to handle multivolumes
        raise NotImplementedError

    def map_scene(self, S):
        raise NotImplementedError

    def get_mapping(self, obj):
        match obj:
            case BaseGenerator():
                self.mapping = self.map_generator(obj)
            case BaseVolume():
                self.mapping = self.map_volume(obj)
            case BaseScene():
                self.mapping = self.map_scene(obj)
            case _:
                raise TypeError()
            
    def inverse_map_generator(self, node: BaseGeneratorNode) -> NullGenerator | Generator | AmorphousGenerator:
        params = {**node}
        children = params.pop('children')
        if len(children) != 0:
            raise ValueError("Generator Nodes should not have children.")
        
        match node:
            case NullGeneratorNode() | AmorphousGeneratorNode():
                return node.class_type(**params)
            case GeneratorNode():
                lattice = Lattice(params.pop('lattice_matrix'))
                species = params.pop('basis_species')
                coords = params.pop('basis_coords')
                structure = Structure(lattice, species, coords)
                return Generator(structure=structure, **params)
            case BaseGeneratorNode():
                raise TypeError("Base Generators should not be constructed directly.")
            case _:
                raise TypeError

    
    def to_object(self):
        match self.mapping:
            case BaseGeneratorNode():
                return self.inverse_map_generator(self.mapping)
            case _:
                raise NotImplementedError