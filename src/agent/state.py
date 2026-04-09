from __future__ import annotations

from typing import Annotated, List, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int
    rewritten_question: Optional[str]
    tool_used: Optional[str]
    web_results: Optional[str]
    grade: Optional[str]
    session_id: Optional[str]
    episodic_context: Optional[str]
    messages: Optional[List[BaseMessage]]