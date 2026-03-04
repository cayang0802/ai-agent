from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from vectordb.interface import VectorStoreInterface

logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    debug_retrieval: bool = False

_RAG_TEMPLATE = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are a helpful assistant. Use the following document excerpts to answer "
        "the user's question. If the excerpts do not contain enough information, say so.\n\n"
        "--- Document Context ---\n"
        "{context}\n"
        "--- End of Context ---\n\n"
        "User question: {query}\n\n"
        "Answer:"
    ),
)


class RAGEngine:
    def __init__(self, store: VectorStoreInterface, llm, debug: bool = False) -> None:
        self._store = store
        self._llm = llm
        self._debug = debug

    def retrieve(self, query: str, k: int) -> list[Document]:
        chunks = self._store.similarity_search(query, k=k)
        if self._debug:
            logger.info("==== RAG Retrieval Debug ====")
            logger.info("Query: %s", query)
            logger.info("Retrieved %d chunk(s):", len(chunks))
            for i, doc in enumerate(chunks, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                logger.info("  [%d] source=%s page=%s | %s", i, source, page, doc.page_content[:200])
            logger.info("=============================")
        else:
            logger.debug("RAG retrieve: query=%r → %d chunks", query, len(chunks))
        return chunks

    def augment(self, query: str, chunks: list[Document]) -> str:
        parts: list[str] = []
        for i, doc in enumerate(chunks, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
        context = "\n\n".join(parts)
        return _RAG_TEMPLATE.format(context=context, query=query)

    def generate(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return response.content

    def run(self, query: str, k: int = 1) -> str:
        chunks = self.retrieve(query, k=k)
        if not chunks:
            logger.info("RAG: no chunks found for query=%r, skipping", query)
            return ""
        prompt = self.augment(query, chunks)
        result = self.generate(prompt)
        logger.info("RAG generate: %d chars", len(result))
        return result
