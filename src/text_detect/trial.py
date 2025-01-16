from loguru import logger

logger.add("reports/logs/my_log.log", level="DEBUG", rotation="100 MB")

logger.debug("Used for debugging your code.")