import logging
import sys

# Define the log format to include timestamp, log level, component name, and message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(component_name: str) -> logging.Logger:
    """
    Creates or retrieves a logger for a specific component.
    Ensures that handlers are not duplicated if called multiple times.
    """
    logger = logging.getLogger(component_name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to prevent duplicate logging
    if not logger.handlers:
        # Prevent log messages from propagating to the root logger
        # which might have its own handlers, causing duplicates.
        logger.propagate = False
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
