from __future__ import annotations

from langchain_core.tools import tool

_rag_engine = None


def init_rag_engine(engine) -> None:
    global _rag_engine
    _rag_engine = engine


@tool
def get_rag_result(query: str, k: int = 1) -> str:
    """Search indexed documents and return relevant information for the given query.

    Args:
        query: The question or topic to search for in the indexed documents.
        k: Number of document chunks to retrieve (default 1).
    """
    if _rag_engine is None:
        return "RAG engine is not initialized."
    return _rag_engine.run(query, k=k)
