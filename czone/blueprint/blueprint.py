from dataclasses import dataclass, field

from abc import ABC, abstractmethod
from collections.abc import Mapping

from czone.util.eset import EqualSet
from czone.volume import BaseVolume, BaseAlgebraic, Volume, MultiVolume, Voxel, Plane, Sphere, Cylinder
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
    children: list = field(default_factory=list)

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
    def add_node(self, node):
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
    
    def add_node(self, node):
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
    
class BaseAlgebraicNode(BaseNode):

    @property
    def is_leaf(self):
        return True
    
    def add_node(self, node):
        pass

@dataclass
class BaseVolumeNode(BaseNode):
    priority: int

    @property
    def base_type(self):
        return BaseVolume
    
    @property
    def class_type(self):
        return BaseVolume

    @property
    def is_leaf(self):
        return False

@dataclass    
class VolumeNode(BaseVolumeNode):
    tolerance: float
    points: np.ndarray | None
    
    def add_node(self, node):
        match node:
            case BaseGeneratorNode():
                ## check if there are any Generators in Children
                is_first_generator = True
                for i, c in enumerate(self.children):
                    if isinstance(c, BaseGeneratorNode):
                        is_first_generator = False
                        break

                if is_first_generator:
                    self.children.append(node)
                else:
                    self.children[i] = node

            case BaseAlgebraicNode():
                self.children.append(node)
            case _:
                raise TypeError("VolumeNodes can only be parent to BaseGeneratorNodes")

class MultiVolumeNode(BaseVolumeNode):
    def add_node(self, node):
        match node:
            case VolumeNode() | MultiVolumeNode():
                self.children.append(node)
            case _:
                raise TypeError("MultiVolumeNodes can only be parent to other MultiVolumes or Volumes")

class BaseSceneNode(BaseNode):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

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
        
    ####################
    # Forward mappings #
    ####################

    def get_mapping(self, obj):
        self.mapping = self.forward_map(obj)

    @staticmethod
    def forward_map(obj):
        match obj:
            case BaseGenerator():
               return Blueprint.map_generator(obj)
            case BaseVolume():
               return Blueprint.map_volume(obj)
            case BaseScene():
               return Blueprint.map_scene(obj)
            case _:
                raise TypeError()

    @staticmethod
    def map_generator(G: BaseGenerator):
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
                raise TypeError

    @staticmethod
    def map_algebraic(A: BaseAlgebraic):
        raise NotImplementedError
            
    @staticmethod
    def map_volume(V):
        # Should be recursive, to handle multivolumes
        match V:
            case MultiVolume():
                node = MultiVolumeNode(V.priority)
                for o in V.volumes:
                    node.add_node(Blueprint.map_volume(o))
            case Volume():
                node = VolumeNode(priority=V.priority, tolerance=V.tolerance, points=V.points)
                node.add_node(Blueprint.map_generator(V.generator))
                for o in V.alg_objects:
                    node.add_node(Blueprint.map_algebraic(o))
            case _:
                raise TypeError

        return node

    @staticmethod
    def map_scene(S):
        raise NotImplementedError

    ####################
    # Inverse mappings #
    ####################
    
    def to_object(self):
        return self.inverse_map(self.mapping)
    
    @staticmethod
    def inverse_map(node):
        match node:
            case BaseGeneratorNode():
                return Blueprint.inverse_map_generator(node)
            case BaseAlgebraicNode():
                return Blueprint.inverse_map_algebraic(node)
            case BaseVolumeNode():
                return Blueprint.inverse_map_volume(node)
            case BaseSceneNode():
                return Blueprint.inverse_map_scene(node)
            case _:
                raise TypeError

    @staticmethod
    def inverse_map_generator(node: BaseGeneratorNode) -> NullGenerator | Generator | AmorphousGenerator:
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

    @staticmethod
    def inverse_map_algebraic(node: BaseAlgebraicNode) -> Plane | Sphere | Cylinder:
        raise NotImplementedError

    @staticmethod
    def inverse_map_volume(node: BaseVolumeNode) -> Volume | MultiVolume:
        params = {**node}
        children = params.pop('children')
        match node:
            case MultiVolumeNode():
                volumes = [Blueprint.inverse_map_volume(c) for c in children]
                return MultiVolume(volumes, **params)
            case VolumeNode():
                params['alg_objects'] = []
                for c in children:
                    if isinstance(c, BaseAlgebraicNode):
                        params['alg_objects'].append(Blueprint.inverse_map_algebraic(c))
                    else:
                        params['generator'] = Blueprint.inverse_map_generator(c)
                return Volume(**params)
            case BaseVolumeNode():
                raise TypeError("Base Volumes should not be constructed directly")
            case _:
                raise TypeError

    @staticmethod
    def inverse_map_scene(node: BaseSceneNode) -> Scene | PeriodicScene:
        raise NotImplementedError