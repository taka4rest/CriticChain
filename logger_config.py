import logging
import os
from logging.handlers import TimedRotatingFileHandler
import datetime

def setup_logger(name="app_logger", log_dir="logs"):
    """
    Sets up a logger with daily rotation.
    """
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Check if handler already exists to avoid duplicate logs
    if not logger.handlers:
        # Daily rotation at midnight
        log_file = os.path.join(log_dir, "app.log")
        handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        
        # Formatter
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # Also log to console for dev visibility
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def log_token_usage(node_name, model_name, input_tokens, output_tokens, log_dir="logs", session_id=None):
    """
    Logs token usage to a CSV file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    csv_file = os.path.join(log_dir, "token_usage.csv")
    file_exists = os.path.isfile(csv_file)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tokens = input_tokens + output_tokens
    
    # Handle session_id
    session_str = str(session_id) if session_id else "N/A"
    
    with open(csv_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Timestamp,Node,Model,InputTokens,OutputTokens,TotalTokens,SessionId\n")
        f.write(f"{timestamp},{node_name},{model_name},{input_tokens},{output_tokens},{total_tokens},{session_str}\n")
