from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectordb.interface import VectorStoreInterface


class Indexer:
    def add_file_to_db(self, file_path: str) -> int:
        raise NotImplementedError


class PDFIndexer(Indexer):
    def __init__(
        self,
        store: VectorStoreInterface,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def add_file_to_db(self, file_path: str) -> int:
        docs = PyPDFLoader(file_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        ).split_documents(docs)
        self._store.add_documents(chunks)
        return len(chunks)
