import pickle
import hashlib
from pathlib import Path

from src.util.constants import Directory


def pickle_cache(ignore_caching: bool, cachedir: str = 'tmp'):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):

        def wrapped(*args, **kwargs):

            if ignore_caching:
                return fn(*args, **kwargs)

            # init directory for caching
            cache_dir = Directory.OUTPUT_DIR / cachedir
            cache_dir.mkdir(exist_ok=True, parents=True)

            # create filename from kwargs
            filename_components = [fn.__code__.co_filename, fn.__name__]
            filename_components += [str(arg) for arg in args]
            filename_components += [f"{k}_{v}" for k, v in kwargs.items()]
            filename_verbose = "__".join(filename_components)

            # create hash for shorter filenames
            hash_object = hashlib.sha1(str.encode(filename_verbose))
            filename_pickle = cache_dir / f"{hash_object.hexdigest()}.pickle"

            # if cache exists -> load it and return its content
            if Path(filename_pickle).exists():
                with open(filename_pickle, 'rb') as cachehandle:
                    return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(filename_pickle, 'wb') as cachehandle:
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator