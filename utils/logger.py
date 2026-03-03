import logging
import os
import sys

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "agent.log"
    format: str = "%(asctime)s %(levelname)s:%(name)s:%(message)s"


class ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG:    "\033[36m",    # cyan
        logging.INFO:     "\033[32m",    # green
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self._COLORS.get(record.levelno, "")
        return f"{color}{msg}{self._RESET}" if color else msg


def setup_logging(cfg: LoggingConfig, base_dir: str) -> None:
    """每次啟動主程式時清空 agent.log，並將後續 log 寫入檔案與主終端。"""
    root_logger = logging.getLogger()
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    plain_fmt = logging.Formatter(cfg.format)
    log_file = os.path.join(base_dir, cfg.file)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(plain_fmt)

    color_fmt = ColorFormatter(cfg.format) if sys.stdout.isatty() else plain_fmt
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(color_fmt)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
