from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from core.agent import Agent, AgentConfig
from frontend.app import ChatApp
from utils import setup_logging
from utils.logger import LoggingConfig


class AppConfig(BaseModel):
    logging: LoggingConfig = LoggingConfig()
    agent: AgentConfig = AgentConfig()


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
    ChatApp(agent, debug_tool_calls=debug).run()


if __name__ == "__main__":
    main()
