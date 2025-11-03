import os
import sys
import logging

logging_st = "[%(asctime)s] %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "current_log.log")

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_st,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Classifier")
logger.info("Logging system initialized")

__all__ = ["logger"]