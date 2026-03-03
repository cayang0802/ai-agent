from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vectordb.interface import VectorStoreInterface


class ChromaVectorStore(VectorStoreInterface):
    def __init__(self, persist_directory: str, embeddings) -> None:
        self._db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    def add_documents(self, documents: list[Document]) -> None:
        self._db.add_documents(documents)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._db.similarity_search(query, k=k)

    def clear(self) -> None:
        self._db.delete_collection()
