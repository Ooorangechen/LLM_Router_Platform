# scr/utilis/logger.py
# public function: setup_logging(), get_logger()


import logging
import logging.handlers
import sys
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

NAMESPACE = "llm_router"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "logs/llm_router.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

TEXT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
JSON_FORMAT = ("%(asctime)s %(name)s %(levelname)s %(message)s "
               "%(process)d %(thread)d %(module)s %(lineno)d")

def setup_logging(log_level=DEFAULT_LOG_LEVEL, 
                  log_file=DEFAULT_LOG_FILE, 
                  console_output=True, 
                  structured_logs=True,
                  max_bytes= DEFAULT_MAX_BYTES, 
                  backup_count=DEFAULT_BACKUP_COUNT):
    logger = logging.getLogger(NAMESPACE)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False # avoid repeating output

    # remove existing handlers if there's any
    for old in list(logger.handlers):
        logger.removeHandler(old)
        old.close()

    text_formatter = logging.Formatter(TEXT_FORMAT)

    try:
        # create parent if not exist, don't raise error if log file exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(text_formatter)
            logger.addHandler(console_handler)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count,
            encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(text_formatter)
        logger.addHandler(file_handler)

        if structured_logs:
            json_handler = logging.handlers.RotatingFileHandler(
                f"{log_file}.jsonl", maxBytes=max_bytes,
                backupCount=backup_count, encoding="utf-8")
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(JsonFormatter(JSON_FORMAT))
            logger.addHandler(json_handler)

    except Exception:
        for broken in list(logger.handlers):
            logger.removeHandler(broken)
            broken.close()
        fallback_handler = logging.StreamHandler(sys.stderr)
        fallback_handler.setLevel(logging.WARNING)
        fallback_handler.setFormatter(text_formatter)
        logger.addHandler(fallback_handler)
        logger.warning("Logging setup failed: degraded to console Warning only.")

    return logger


def get_logger(name:str) -> logging.Logger:
    namespace_logger = logging.getLogger(NAMESPACE)
    if not namespace_logger.handlers:
        namespace_logger.addHandler(logging.NullHandler())
    return logging.getLogger(f"{NAMESPACE}.{name}")