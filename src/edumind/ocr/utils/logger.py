import sys

from loguru import logger

from ..config import LOG_FORMAT, LOG_LEVEL, LOGS_DIR

# Remove default handler
logger.remove()

# Add console handler
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True,
)

# Add file handler
logger.add(
    LOGS_DIR / "data_ingestion_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    rotation="1 day",
    retention="7 days",
    compression="zip",
)

def get_logger(name: str):
    """Get a logger instance with a specific name."""
    return logger.bind(name=name)
