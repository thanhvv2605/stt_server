from typing import List, Optional

from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class Word(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionJSON(BaseModel):
    text: str


class TranscriptionVerboseJSON(BaseModel):
    text: str
    language: str
    duration: float
    segments: List[Segment]
    words: Optional[List[Word]] = None
