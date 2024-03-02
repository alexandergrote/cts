from abc import ABC, abstractmethod


class BasePreprocessor(ABC):

    @abstractmethod
    def execute(self, *args, **kwargs) -> dict:
        raise NotImplementedError