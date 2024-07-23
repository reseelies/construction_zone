from abc  import ABC, abstractmethod
from czone.blueprint import Blueprint
from pathlib import Path

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
        cls.deserialize(filepath, **kwargs)


class h5_Serializer(BaseSerializer):

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        raise NotImplementedError

class json_Serializer(BaseSerializer):

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        raise NotImplementedError

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


    