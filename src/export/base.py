from abc import ABC, abstractmethod

class BaseExporter(ABC):

    @abstractmethod
    def export(self) -> dict:
        raise NotImplementedError