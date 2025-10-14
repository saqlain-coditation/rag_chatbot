from typing import Optional

from llama_index.core.workflow import Event


class JudgeQueryEvent(Event):
    query: str


class ImproveQueryEvent(Event):
    query: str


class AttachContextEvent(Event):
    query: str


class SearchEvent(Event):
    query: str


class JudgeResponseEvent(Event):
    query: str
    response: str
    data: Optional[str] = None


class ReQueryEvent(Event):
    query: str
    response: str


class AnswerEvent(Event):
    query: str
    response: str
