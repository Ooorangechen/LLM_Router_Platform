# scr/utilis/logger.py
# public function: setup_logging(), get_logger()


import logging
import logging.handlers
import sys
from pathlib import Path

# from pythonjsonlogger.json import JsonFormatter

##### Default Configs #####

NAMESPACE = "llm_router"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "logs/llm_router.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

TEXT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
JSON_FORMAT = ("%(asctime)s %(name)s %(levelname)s %(message)s "
               "%(process)d %(thread)d %(module)s %(lineno)d")

########################


def setup_logging(log_level=DEFAULT_LOG_LEVEL,
                  log_file=DEFAULT_LOG_FILE,
                  console_output=True,
                  structured_logs=True,
                  max_bytes= DEFAULT_MAX_BYTES,
                  backup_count=DEFAULT_BACKUP_COUNT,
                  log_format=TEXT_FORMAT,
                  json_format=JSON_FORMAT):
    logger = logging.getLogger(NAMESPACE)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False # avoid repeating output

    # remove existing handlers if there's any
    for old in list(logger.handlers):
        logger.removeHandler(old)
        old.close()

    text_formatter = logging.Formatter(log_format)
    failures = []

    # console first: it needs nothing from the filesystem, so a disk or permission
    # problem can no longer cost us the one channel that would have worked
    if console_output:
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(text_formatter)
            logger.addHandler(console_handler)
        except Exception as exc:
            failures.append(f"console channel: {exc}")

    log_dir_ready = True
    try:
        # create parent if not exist, don't raise error if log file exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log_dir_ready = False
        failures.append(f"log directory: {exc}")

    if log_dir_ready:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count,
                encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(text_formatter)
            logger.addHandler(file_handler)
        except Exception as exc:
            failures.append(f"file channel: {exc}")

        if structured_logs:
            try:
                from pythonjsonlogger.json import JsonFormatter      # make sure not fail when main.py setup 
                json_handler = logging.handlers.RotatingFileHandler(
                    f"{log_file}.jsonl", maxBytes=max_bytes,
                    backupCount=backup_count, encoding="utf-8")
                json_handler.setLevel(logging.INFO)
                json_handler.setFormatter(JsonFormatter(json_format))
                logger.addHandler(json_handler)
            except ImportError:
                logger.debug("python-json-logger not installed; skipping JSON log channel")
            except Exception as exc:
                failures.append(f"structured channel: {exc}")

    # 3.3.7: always keep at least one StreamHandler so the service never dies from
    # logging. Only reached when every channel above failed or was switched off.
    if not logger.handlers:
        fallback_handler = logging.StreamHandler(sys.stderr)
        fallback_handler.setLevel(logging.WARNING)
        fallback_handler.setFormatter(text_formatter)
        logger.addHandler(fallback_handler)
        logger.warning("Logging setup failed: degraded to console Warning only.")

    # reported after the handlers exist, so these warnings are actually visible
    for failure in failures:
        logger.warning(f"Logging channel unavailable, {failure}")

    return logger


def get_logger(name:str) -> logging.Logger:
    namespace_logger = logging.getLogger(NAMESPACE)
    if not namespace_logger.handlers:
        namespace_logger.addHandler(logging.NullHandler())
    return logging.getLogger(f"{NAMESPACE}.{name}")