from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from core.agent import Agent, AgentConfig
from rag.evaluator import RAGEvaluator, RAGEvaluatorConfig
from rag.rag_engine import RAGConfig, RAGEngine
from frontend.app import ChatApp
from model.embedding import EmbeddingConfig, EmbeddingFactory
from model.llm import LLMFactory
from rag.indexer import PDFIndexer, TXTIndexer
from tools.rag import init_rag_engine, init_rag_evaluator
from utils import setup_logging
from utils.logger import LoggingConfig
from vectordb.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


class AppConfig(BaseModel):
    logging: LoggingConfig = LoggingConfig()
    agent: AgentConfig = AgentConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    rag: RAGConfig = RAGConfig()
    evaluator: RAGEvaluatorConfig = RAGEvaluatorConfig()


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
    indexers = {
        ".pdf": PDFIndexer(store),
        ".txt": TXTIndexer(store),
    }
    rag_engine = RAGEngine(store=store, llm=llm, debug=cfg.rag.debug_retrieval)
    init_rag_engine(rag_engine)

    if cfg.evaluator.enabled:
        try:
            from langfuse import Langfuse
            langfuse = Langfuse()
            evaluator = RAGEvaluator(langfuse=langfuse, llm=llm, embeddings=embeddings)
            init_rag_evaluator(evaluator)
            logger.info("RAG evaluator enabled (Langfuse).")
        except Exception:
            logger.exception("Failed to initialize RAG evaluator; evaluation disabled.")

    ChatApp(agent, indexers=indexers, debug_tool_calls=debug).run()


if __name__ == "__main__":
    main()
