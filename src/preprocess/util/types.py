from pydantic import BaseModel
from typing import List, Optional, Union

from src.preprocess.util.metrics import EntropyCalculator

class AnnotatedSequence(BaseModel):
    id_value: str
    sequence_values: List[str]
    class_value: Optional[int] = None


class FrequentPattern(BaseModel):

    sequence_values: List[str]

    support: int
    support_pos: Optional[int] = None
    support_neg: Optional[int] = None

    @property
    def inverse_entropy(self) -> float:

        if self.support_pos is None:
            raise ValueError("Inverse entropy calculation not possible")

        entropy = EntropyCalculator.calculate_entropy(
            probability= self.support_pos / self.support
        )

        return 1 - entropy


class FrequentPatternWithConfidence(BaseModel):

    antecedent: List[str]
    consequent: List[str]

    support: int
    support_pos: Optional[int] = None
    support_neg: Optional[int] = None

    confidence: float
    confidence_pos: Optional[float] = None
    confidence_neg: Optional[float] = None

    @property
    def delta_confidence(self) -> float:

        if (self.confidence_pos is None) or (self.confidence_neg is None):
            raise ValueError("Delta confidence calculation not possible.")

        return self.confidence_pos - self.confidence_neg
    
    @property
    def inverse_entropy(self) -> float:

        if self.support_pos is None:
            raise ValueError("Delta confidence calculation not possible.")

        entropy = EntropyCalculator.calculate_entropy(
            probability= self.support_pos / self.support
        )

        return 1 - entropy


class StackObject(BaseModel):
    database: List[List[str]]
    prefix: List[str]
    classes: Union[List[int], List[None]]

    @classmethod
    def from_annotated_sequences(cls, annotated_sequences: List[AnnotatedSequence], prefix: List[str]):
        
        database = []
        classes = []

        for annotated_sequence in annotated_sequences:
            database.append(annotated_sequence.sequence_values)
            classes.append(annotated_sequence.class_value)
        
        return cls(prefix=prefix, database=database, classes=classes)
