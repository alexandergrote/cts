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
        if 'cached_functions' in data:
            for function in data['cached_functions']:
                os.environ[function] = "cached"

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