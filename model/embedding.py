from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"
    base_url: str | None = None   # set for on-premises OpenAI-compatible servers


class EmbeddingFactory:
    @staticmethod
    def create(config: EmbeddingConfig) -> OpenAIEmbeddings:
        kwargs = {"model": config.model}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAIEmbeddings(**kwargs)
