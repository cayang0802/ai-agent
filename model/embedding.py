from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"


class EmbeddingFactory:
    @staticmethod
    def create(config: EmbeddingConfig) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model=config.model)
