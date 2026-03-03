import os

from pydantic import BaseModel, Field
from langchain.agents import create_agent

from tools import TOOLS
from model.llm import LLMConfig, LLMFactory


class AgentConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    system_prompt: str = "你是一個工具型助理，請用繁體中文回覆。"
    # 未來 RAG：retrievers: list = []


class Agent:
    def __init__(self, config: AgentConfig):
        self._config = config

    def build(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "找不到 OPENAI_API_KEY。請在專案根目錄建立 .env，內容例如：OPENAI_API_KEY=你的金鑰"
            )
        llm = LLMFactory.create(self._config.llm)
        return create_agent(
            model=llm,
            tools=TOOLS,
            system_prompt=self._config.system_prompt,
        )
