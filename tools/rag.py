from __future__ import annotations

import asyncio
import logging
import threading

from langchain_core.tools import tool

from rag.evaluator import RAGEvaluator
from rag.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

_rag_engine: RAGEngine | None = None
_evaluator: RAGEvaluator | None = None


def init_rag_engine(engine: RAGEngine) -> None:
    global _rag_engine
    _rag_engine = engine


def init_rag_evaluator(evaluator: RAGEvaluator) -> None:
    global _evaluator
    _evaluator = evaluator


def _run_evaluation(evaluator: RAGEvaluator, query: str, contexts: list[str], answer: str) -> None:
    """Run RAG evaluation in a dedicated event loop (RAGAS may use async internally)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        evaluator.ragas_evaluate(query, contexts, answer)
    except Exception:
        logger.exception("RAG evaluation failed")
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@tool
def get_rag_result(query: str, k: int = 3) -> str:
    """Search indexed documents and return relevant information for the given query.

    Args:
        query: The question or topic to search for in the indexed documents.
        k: Number of document chunks to retrieve (default 3).
    """
    if _rag_engine is None:
        return "RAG engine is not initialized."
    answer, contexts = _rag_engine.run(query, k=k)
    if _evaluator is not None and contexts is not None:
        t = threading.Thread(
            target=_run_evaluation,
            args=(_evaluator, query, contexts, answer),
            daemon=False,
        )
        t.start()
    return answer
