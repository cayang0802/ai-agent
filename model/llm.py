from __future__ import annotations
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: Literal["openai", "vllm", "llamacpp"] = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None


def _build_openai_or_compatible(config: LLMConfig) -> ChatOpenAI:
    kwargs = {"model": config.model}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)


class LLMFactory:
    @staticmethod
    def create(config: LLMConfig):
        if config.provider in ("openai", "vllm", "llamacpp"):
            return _build_openai_or_compatible(config)
        raise ValueError(f"Unknown provider: {config.provider}")
