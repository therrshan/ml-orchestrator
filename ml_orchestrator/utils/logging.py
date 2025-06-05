"""
Logging configuration for the ML orchestrator.
"""

import logging
import sys
from typing import Optional
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        # Save original format
        original_format = self._style._fmt
        
        # Add color to the level name
        if record.levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
            record.levelname = colored_levelname
        
        # Format the record
        formatted = super().format(record)
        
        # Restore original format
        self._style._fmt = original_format
        
        return formatted


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the orchestrator.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ml_orchestrator")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ml_orchestrator") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)