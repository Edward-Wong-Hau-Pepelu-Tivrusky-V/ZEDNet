import logging
import sys

def setup_logger(log_file="pipeline.log", clear_log=False):
    """
    clear_log=True  → wipes log file each run
    clear_log=False → appends to existing log
    """

    logger = logging.getLogger("pipeline_logger")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- File handler ---
    mode = "w" if clear_log else "a"
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.DEBUG)

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # --- Formatting ---
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger