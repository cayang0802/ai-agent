from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vectordb.interface import VectorStoreInterface


def _tokenize(text: str) -> list[str]:
    """Character-level CJK + word-level Latin tokenizer for BM25."""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())



class ChromaVectorStore(VectorStoreInterface):
    def __init__(self, persist_directory: str, embeddings) -> None:
        self._db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        self._bm25_corpus: list[Document] | None = None  # lazy cache, cleared on add

    def add_documents(self, documents: list[Document]) -> None:
        self._db.add_documents(documents)
        self._bm25_corpus = None  # invalidate BM25 cache

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._db.similarity_search(query, k=k)

    def _load_corpus(self) -> list[Document]:
        """Load all documents from the collection into a local cache for BM25."""
        if self._bm25_corpus is None:
            raw = self._db._collection.get(include=["documents", "metadatas"])
            self._bm25_corpus = [
                Document(page_content=content, metadata=meta or {})
                for content, meta in zip(raw["documents"], raw["metadatas"])
                if content
            ]
        return self._bm25_corpus

    def hybrid_search(self, query: str, k: int, filter: dict | None = None) -> list[Document]:
        """Hybrid search: dense vector (semantic) + sparse vector (BM25) fused via EnsembleRetriever (weighted RRF)."""
        from langchain_community.retrievers import BM25Retriever
        from langchain_classic.retrievers import EnsembleRetriever

        fetch_k = max(k * 4, 20)

        # Dense retriever (semantic)
        vector_retriever = self._db.as_retriever(
            search_kwargs={"k": fetch_k, "filter": filter}
        )

        # Sparse retriever (BM25) — filter corpus in Python, then build retriever
        corpus = self._load_corpus()
        if filter:
            corpus = [
                doc for doc in corpus
                if all(doc.metadata.get(fk) == fv for fk, fv in filter.items())
            ]
        if not corpus:
            return vector_retriever.invoke(query)[:k]

        bm25_retriever = BM25Retriever.from_documents(
            corpus, preprocess_func=_tokenize, k=fetch_k
        )

        # EnsembleRetriever: weighted RRF (equal weights, c=60 built-in)
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        return ensemble.invoke(query)[:k]

    def clear(self) -> None:
        self._db.delete_collection()
        self._bm25_corpus = None
