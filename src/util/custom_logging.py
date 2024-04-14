import time
import yaml

from pickle import dump, load
from pathlib import Path

console = None

try:
    from rich.console import Console
    console = Console()
except ImportError:
    pass


class Pickler:

    @staticmethod
    def write(obj, filename: str):
        with open(filename, 'wb') as file:
            dump(obj, file)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as file:
            data = load(file)

        return data


def log_time(key: str, filename: Path = Path('time.yaml')):

    def decorator(func):

        def wrapped(*args, **kwargs):

            # check if log file already exists
            # load file if it does
            logging_dict = {}
            if filename.exists():
                with open(filename, 'r') as file:
                    logging_dict = yaml.safe_load(file)

            # execute function
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            time_taken = end_time - start_time

            if logging_dict is None:
                return result

            # log time taken
            if key in logging_dict:
                logging_dict[key].append(time_taken)
            else:
                logging_dict[key] = [time_taken]

            # save logging_dict to file
            with open(filename, 'w') as file:
                yaml.safe_dump(logging_dict, file)

            return result

        return wrapped

    return decorator


if __name__ == '__main__':

    @log_time(key='tmp')
    def my_fun(a, b):
        time.sleep(1)
        return a + b

    my_fun(1, 2)