from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class VectorStoreInterface(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> list[Document]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
