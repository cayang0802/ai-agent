import os
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectordb.interface import VectorStoreInterface


class Indexer:
    def __init__(
        self,
        store: VectorStoreInterface,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ) -> None:
        self._store = store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _load_docs(self, file_path: str) -> list[Document]:
        raise NotImplementedError

    def add_file_to_db(self, file_path: str) -> int:
        # 從檔名解析作者資訊與副檔名，例如 report_作者_王小明_.pdf → author: 王小明, file_type: .pdf
        filename = os.path.basename(file_path)
        metadata = {}
        if m := re.search(r"_作者_(.+?)_", filename):
            metadata["author"] = m.group(1)
        metadata["file_type"] = os.path.splitext(filename)[1]
        metadata["file_name"] = filename

        chunks = self._splitter.split_documents(self._load_docs(file_path))

        for chunk in chunks:
            chunk.metadata.setdefault("page", -1)
            chunk.metadata.setdefault("year", -1)
            chunk.metadata.update(metadata)

        self._store.add_documents(chunks)
        return len(chunks)


class PDFIndexer(Indexer):
    def _load_docs(self, file_path: str) -> list[Document]:
        docs = PyPDFLoader(file_path).load()
        for doc in docs:
            doc.metadata["page"] += 1  # 1-based
        return docs


class TXTIndexer(Indexer):
    def _load_docs(self, file_path: str) -> list[Document]:
        return TextLoader(file_path, encoding="utf-8").load()
