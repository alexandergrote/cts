import joblib
import hashlib

import pandas as pd

from typing import Any, Union, List
from pydantic import BaseModel


class Hash(BaseModel):

    @staticmethod
    def hash_dataframe(data: Union[pd.DataFrame, pd.Series]) -> str:
        return joblib.hash(data)

    @staticmethod
    def hash_string(string: str) -> str:
        return hashlib.sha1(str.encode(string)).hexdigest()

    @staticmethod
    def hash(obj: Any) -> str:

        if isinstance(obj, pd.DataFrame):
            return Hash.hash_dataframe(obj)

        if isinstance(obj, pd.Series):
            return Hash.hash_dataframe(obj)

        if isinstance(obj, str):
            return Hash.hash_string(obj)

        if obj is None:
            return Hash.hash_string('None')

        if isinstance(obj, int):
            return str(obj.__hash__())

        if isinstance(obj, float):
            return str(obj.__hash__())
        
        if isinstance(obj, BaseModel):

            # pydantic private fields are not included in model dumps
            # there may be other edge cases where this is not true
            # but for pragmatic reasons we will stick to this solutionfor now.

            # todo: investigate if there is a better way to do this.

            public_values = obj.model_dump()
            private_values = obj.__pydantic_private__ if obj.__pydantic_private__ is not None else {}

            pyd_dict = {**public_values, **private_values}

            return Hash.hash_recursive(**pyd_dict)
        
        if hasattr(obj, 'hash'):

            hash_value = obj.hash()

            if not isinstance(hash_value, str):
                raise ValueError(f"Object {obj} has a 'hash' function, but it does not return a string.")
            
            return hash_value

        from IPython import embed; embed()

        raise ValueError(f"Object {obj} cannot be hashed. You may consider defining an explicit 'hash' function.")
    
    @staticmethod
    def hash_dict(dictionary: dict) -> str:  

        items = {k: v for k, v in dictionary.items()}

        return Hash.hash_recursive(**items)
    
    @staticmethod
    def hash_list(mylist: List):
        return Hash.hash_recursive(*mylist)
    
    @staticmethod
    def hash_tuple(mytuple: tuple):
        return Hash.hash_recursive(*mytuple)
    
    @staticmethod
    def hash_recursive(*args, **kwargs) -> str:

        """
        This function is used to create a unique hash from all arguments. 
        
        It first checks if an argument is a dictionary, list, tuple, etc. and a correspondig hashing function is called
        If an argument does not fall into these categories, it is passed to the hash function.
        """

        # todo: refactor this function to be more intuitive and to avoid repetitive code
        
        hash_list = []
        
        for obj in args:

            if isinstance(obj, dict):
                hash_list.append(Hash.hash_dict(obj))
                continue

            if isinstance(obj, list):
                hash_list.append(Hash.hash_list(obj))
                continue

            if isinstance(obj, tuple):
                hash_list.append(Hash.hash_tuple(obj))
                continue

            hash_list.append(Hash.hash(obj))
        
        for key, value in kwargs.items():

            hash_list.append(Hash.hash(key))

            if isinstance(value, dict):
                hash_list.append(Hash.hash_dict(value))
                continue

            if isinstance(value, list):
                hash_list.append(Hash.hash_list(value))
                continue

            if isinstance(value, tuple):
                hash_list.append(Hash.hash_tuple(value))
                continue

            hash_list.append(Hash.hash(value))

        hash_str = '_'.join(hash_list)
        
        return Hash.hash(hash_str)