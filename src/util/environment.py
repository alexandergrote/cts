import os
from pydantic import BaseModel

from src.util.constants import EnvMode


class PydanticEnvironment(BaseModel):

    mode: EnvMode
    cached_functions: list[str] = []

    class Config:
        extra = 'forbid'

    @staticmethod
    def set_environment_variables(data: dict):

        # in accordance with environment pickle cache
        # specify environment variables for caching functions
        key = 'cached_functions'

        cached_functions = data.get(key, [])  # get the list of cached functions or an empty list if not present

        if isinstance(cached_functions, list):  # check if the value is a list
            for function in cached_functions:
                os.environ[function] = "cached"
        else:
            # handle the case where 'cached_functions' is not a list
            # you might want to log a warning or raise an error here
            raise ValueError(f"'cached_functions' is not a list, got {type(cached_functions)} instead.")

        for key, value in data.items():

            if 'cached_functions' == key:
                continue

            # check if supplied argument is an enum and add value to environment
            if isinstance(value, EnvMode):
                value = value.value

            os.environ[key] = str(value)

    @classmethod
    def create_from_environment(cls):

        # create dictionary with all attributes of the class as keys and its environment variables as values
        data = {key: os.environ[key] for key in cls.model_fields.keys() if key != "cached_functions"}

        # check if attribute is enum and convert it to enum
        for key, value in data.items():
            if cls.__annotations__[key] == EnvMode:
                data[key] = EnvMode(value)

        return cls(**data)
    

if __name__ == '__main__':

    env = PydanticEnvironment(mode=EnvMode.DEV)
    env.set_environment_variables(env.model_dump())
    env = PydanticEnvironment.create_from_environment()

    print(env.model_dump())