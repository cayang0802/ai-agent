from __future__ import annotations

import json
import logging
import re

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from vectordb.interface import VectorStoreInterface

logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    debug_retrieval: bool = False
    reranker_model: str = ""  # empty = disabled

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
    def __init__(self, store: VectorStoreInterface, llm, debug: bool = False,
                 reranker_model: str = "") -> None:
        self._store = store
        self._llm = llm
        self._debug = debug
        self._reranker = CrossEncoder(reranker_model) if reranker_model else None

    def _expand_queries(self, query: str, n: int) -> list[str]:
        if n <= 1:
            return [query]
        prompt = (
            f"Generate {n - 1} different ways to express the following question "
            f"to improve document retrieval. Return ONLY a JSON array of strings.\n\n"
            f"Question: {query}"
        )
        response = self._llm.invoke(prompt)
        try:
            text = re.search(r"\[.*\]", response.content, re.DOTALL).group()
            expanded = json.loads(text)[: n - 1]
        except Exception:
            expanded = []
        return [query] + expanded

    def retrieve(self, query: str, k: int, query_expand_n: int = 2,
                 k_of_n: int = 5, author: str = "") -> list[Document]:
        # Step 1: Query Expansion
        queries = self._expand_queries(query, query_expand_n) if query_expand_n > 1 else [query]

        # Step 2 & 3: Pre-filter + hybrid_search for each query
        filter_dict = {"author": author} if author else None
        seen: set[str] = set()
        all_chunks: list[Document] = []
        for q in queries:
            for doc in self._store.hybrid_search(q, k=k_of_n, filter=filter_dict):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_chunks.append(doc)

        # Step 4: Rerank → top-k
        if self._reranker is None or len(all_chunks) <= k:
            top_chunks = all_chunks[:k]
        else:
            pairs = [(query, doc.page_content) for doc in all_chunks]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(scores, all_chunks), key=lambda x: x[0], reverse=True)
            top_chunks = [doc for _, doc in ranked[:k]]

        if self._debug:
            logger.info("==== RAG Retrieval Debug ====")
            logger.info("Query: %s", query)
            logger.info("Expanded queries: %d, candidates: %d, top-k: %d", len(queries), len(all_chunks), len(top_chunks))
            for i, q in enumerate(queries):
                logger.info("  query[%d]: %s", i, q)
            for i, doc in enumerate(top_chunks, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                author = doc.metadata.get("author", "?")
                logger.info("\n  [%d] source=%s @ page=%s, author=%s\n%s", i, source, page, author, doc.page_content[:200])
            logger.info("=============================")
        else:
            logger.debug("RAG retrieve: query=%r → %d chunks", query, len(top_chunks))
        return top_chunks

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

    def run(self, query: str, k: int = 3) -> tuple[str, list[str] | None]:
        chunks = self.retrieve(query, k=k)
        if not chunks:
            logger.info("RAG: no chunks found for query=%r, skipping", query)
            return "", None
        prompt = self.augment(query, chunks)
        result = self.generate(prompt)
        logger.info("RAG generate: %d chars", len(result))
        contexts = [doc.page_content for doc in chunks]
        return result, contexts
