from pydantic import BaseModel
from typing import Any, Optional


class Rule(BaseModel):

    """

    ('A' -> 'B') -> 'C'

    """

    antecedent: Any
    precedent: str

    # metrics
    support: Optional[float] = None
    confidence: Optional[float] = None