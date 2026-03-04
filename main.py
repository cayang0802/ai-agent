from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from core.agent import Agent, AgentConfig
from rag.rag_engine import RAGConfig, RAGEngine
from frontend.app import ChatApp
from model.embedding import EmbeddingConfig, EmbeddingFactory
from model.llm import LLMFactory
from rag.indexer import PDFIndexer
from tools.rag import init_rag_engine
from utils import setup_logging
from utils.logger import LoggingConfig
from vectordb.chroma import ChromaVectorStore


class AppConfig(BaseModel):
    logging: LoggingConfig = LoggingConfig()
    agent: AgentConfig = AgentConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    rag: RAGConfig = RAGConfig()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    return AppConfig.model_validate(data)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    load_dotenv()
    cfg = load_config()
    setup_logging(cfg.logging, _THIS_DIR)
    debug = os.getenv("DEBUG_TOOL_CALLS", "1") != "0"

    agent = Agent(cfg.agent).build()

    llm = LLMFactory.create(cfg.agent.llm)
    embeddings = EmbeddingFactory.create(cfg.embedding)
    store = ChromaVectorStore(
        persist_directory=os.path.join(_THIS_DIR, "vectordb", "chroma_data"),
        embeddings=embeddings,
    )
    indexer = PDFIndexer(store)
    rag_engine = RAGEngine(store=store, llm=llm, debug=cfg.rag.debug_retrieval)
    init_rag_engine(rag_engine)

    ChatApp(agent, indexer=indexer, debug_tool_calls=debug).run()


if __name__ == "__main__":
    main()
