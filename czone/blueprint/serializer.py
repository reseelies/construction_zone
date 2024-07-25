from abc  import ABC, abstractmethod
from czone.blueprint.blueprint import (
    Blueprint, 
    BaseNode,
    NodeMap,
)
from pathlib import Path

import numpy as np
import json

class BaseSerializer(ABC):
    
    def __init__():
        pass

    @staticmethod
    @abstractmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        """Take a Blueprint and serialize to disk."""
        pass

    @classmethod
    def write(cls, filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        "Alias for serialize."
        cls.serialize(filepath, blueprint, **kwargs)

    @staticmethod
    @abstractmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        """Take a file and try to return a Blueprint"""
        pass

    @classmethod
    def read(cls, filepath: Path | str, **kwargs) -> Blueprint:
        """Alias for deserialize."""
        return cls.deserialize(filepath, **kwargs)


class h5_Serializer(BaseSerializer):

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        raise NotImplementedError

class json_Serializer(BaseSerializer):

    @staticmethod
    def to_dict(node: BaseNode) -> dict:

        res = {**node}
        for k in res.keys():
            if isinstance(res[k], np.ndarray):
                res[k] = res[k].tolist()
        res['_class_type'] = node.class_type.__name__ # force to be first in sort order
        try:
            children = res.pop('children')
        except KeyError:
            children = []

        if len(children) > 0:
            res['children'] = [json_Serializer.to_dict(n) for n in children]

        return res
    
    @staticmethod
    def from_dict(bdict: dict) -> BaseNode:

        try:
            children = bdict.pop('children')
        except KeyError:
            children = []

        res = NodeMap[bdict.pop('_class_type')](**bdict)
        for n in children:
            res.add_node(json_Serializer.from_dict(n))

        return res

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        bdict = json_Serializer.to_dict(blueprint.mapping)

        try: 
            indent = kwargs.pop('indent')
        except KeyError:
            indent = 4

        try:
            sort_keys = kwargs.pop('sort_keys')
        except KeyError:
            sort_keys = True

        with open(filepath, 'w') as f:
            json.dump(bdict, f, sort_keys=sort_keys, indent=indent, **kwargs)

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:

        with open(filepath, 'r') as f:
            bdict = json.load(f)

        node = json_Serializer.from_dict(bdict)
        return Blueprint(node)

class yaml_Serializer(BaseSerializer):

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        raise NotImplementedError
    
class Serializer(BaseSerializer):
    """Dispatch class."""

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        ## Get format from **kwargs if passed in; otherwise, infer from filepath
        output_format = kwargs.get('format', str(filepath).rsplit('.')[-1])

        match output_format:
            case 'h5' | 'H5' | 'hdf5':
                return h5_Serializer.serialize(filepath, blueprint, **kwargs)
            case 'json':
                return json_Serializer.serialize(filepath, blueprint, **kwargs)
            case 'yaml':
                return yaml_Serializer.serialize(filepath, blueprint, **kwargs)
            case _:
                raise ValueError(f"Unsupported format {output_format} detected or passed in.")

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        ## Get format from **kwargs if passed in; otherwise, infer from filepath
        input_format = kwargs.get('format', str(filepath).rsplit('.')[-1])

        match input_format:
            case 'h5' | 'H5' | 'hdf5':
                return h5_Serializer.deserialize(filepath, **kwargs)
            case 'json':
                return json_Serializer.deserialize(filepath, **kwargs)
            case 'yaml':
                return yaml_Serializer.deserialize(filepath, **kwargs)
            case _:
                raise ValueError(f"Unsupported format {input_format} detected or passed in.")


    