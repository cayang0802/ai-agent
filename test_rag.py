"""Quick smoke-test for hybrid_search via RAGEngine."""
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

load_dotenv()

from langchain_openai import ChatOpenAI

from model.embedding import EmbeddingConfig, EmbeddingFactory
from rag.rag_engine import RAGConfig, RAGEngine
from vectordb.chroma import ChromaVectorStore

QUERY = "國科會開始推動發展「可信任生成式AI 對話引擎」(Trustworthy AI DialogueEngine, TAIDE) 的月份?"

def main():
    data = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    rag_cfg = RAGConfig.model_validate(data.get("rag", {}))
    emb_cfg = EmbeddingConfig.model_validate(data.get("embedding", {}))

    embeddings = EmbeddingFactory.create(emb_cfg)
    store = ChromaVectorStore(persist_directory="vectordb/chroma_data", embeddings=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini")
    engine = RAGEngine(store=store, llm=llm, debug=True, reranker_model=rag_cfg.reranker_model)

    print(f"Query: {QUERY}\n")
    result, _ = engine.run(QUERY, k=3)
    print(f"Answer: {result}")

if __name__ == "__main__":
    main()
