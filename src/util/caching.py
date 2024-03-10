import pandas as pd
import pickle
import hashlib
import joblib
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Any, Optional

from src.util.constants import Directory
from src.util.filepath_converter import normalize_to_posix_path


class PickleCacheHandler(BaseModel):

    filepath: Path

    @field_validator("filepath")
    def _set_directory(cls, v):
        return Directory.OUTPUT_DIR / v

    def read(self) -> Optional[Any]: 

        if not self.filepath.exists():
            return None
        
        with open(self.filepath, 'rb') as cachehandle:
            return pickle.load(cachehandle)
 
    def write(self, obj: Any):

        if not self.filepath.exists():
            self.filepath.parent.mkdir(exist_ok=True, parents=True)

        # write to cache file
        with open(self.filepath, 'wb') as cachehandle:
            pickle.dump(obj, cachehandle)


def hash_dataframe(data: pd.DataFrame) -> str:
    return joblib.hash(data)



def pickle_cache(ignore_caching: bool, cachedir: str = 'tmp'):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):

        def wrapped(*args, **kwargs):

            if ignore_caching:
                return fn(*args, **kwargs)
            
            # create filename from kwargs
            filename_components = [fn.__code__.co_filename, fn.__name__]
            filename_components += [str(arg) for arg in args]
            filename_components += [f"{k}_{v}" for k, v in kwargs.items()]
            filename_verbose = "__".join(filename_components)

            # create hash for shorter filenames
            hash_object = hashlib.sha1(str.encode(filename_verbose))
            filename_pickle = Path(cachedir) / f"{hash_object.hexdigest()}.pickle"

            cache_handler = PickleCacheHandler(
                filepath=filename_pickle
            )

            res = cache_handler.read()

            if res is not None:
                return res

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            cache_handler.write(obj=res)

            return res

        return wrapped

    return decorator