from dataclasses import dataclass

from czone.util.eset import EqualSet
from czone.volume import BaseVolume, Volume, MultiVolume, Voxel, Plane, Sphere, Cylinder
from czone.scene import BaseScene, Scene, PeriodicScene
from czone.generator import BaseGenerator, AmorphousGenerator, Generator, NullGenerator
from czone.molecule import Molecule

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

    def map_generator(self, G):
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
    
    def to_object(self):
        raise NotImplementedError