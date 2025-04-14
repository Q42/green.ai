from pydantic import BaseModel, Field
from typing import List, Optional


class ConversationDescription(BaseModel):
    descriptions: List[str]


class ConversationCoherenceVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Reason(BaseModel):
    reason: str
