from pydantic import BaseModel
from typing import List, Optional

class AnnotatedSequence(BaseModel):
    id_value: str
    sequence_values: List[str]
    class_value: Optional[int] = None