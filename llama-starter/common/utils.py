"""
common/utils.py

Utility functions for logging, ID generation, checksums, and timing.

This module provides shared utilities used across the library.
No external side effects - all functions are pure or self-contained.
"""

import hashlib
import logging
import time
from typing import Optional
from datetime import datetime
import uuid


# Configure default logger
logger = logging.getLogger("llama_rag")
logger.setLevel(logging.WARNING)  # Default: minimal output

if not logger.handlers:
    handler = logging.StreamHandler()
    # Clean format for non-verbose mode
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def configure_logging(verbose: bool = False):
    """
    Configure logging level and format.
    
    Args:
        verbose: If True, enable detailed logging with timestamps.
                 If False, minimal clean output (warnings and errors only).
    
    Example:
        >>> configure_logging(verbose=True)  # Enable detailed logs
        >>> configure_logging(verbose=False) # Clean minimal output
    """
    logger = logging.getLogger("llama_rag")
    
    if verbose:
        logger.setLevel(logging.DEBUG)
        # Detailed format with timestamps for verbose mode
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        logger.setLevel(logging.WARNING)
        # Clean format for non-verbose mode
        formatter = logging.Formatter("%(message)s")
    
    # Update formatter for all handlers
    for handler in logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(f"llama_rag.{name}")


def generate_doc_id(text: str, source: str = "") -> str:
    """
    Generate a deterministic document ID from text and source.
    
    Args:
        text: Document text content
        source: Optional source identifier
    
    Returns:
        Unique document ID in format 'doc_<hash_prefix>'
    
    Design notes:
        - Uses SHA256 for collision resistance
        - Truncated to 12 chars for readability
        - Deterministic: same input -> same ID
    
    Example:
        >>> generate_doc_id("Hello world", "file.txt")
        'doc_a3b2c1d4e5f6'
    """
    content = f"{source}:{text}"
    hash_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"doc_{hash_digest[:12]}"


def generate_unique_id(prefix: str = "id") -> str:
    """
    Generate a unique ID using UUID4.
    
    Args:
        prefix: Optional prefix for the ID
    
    Returns:
        Unique ID string
    
    Example:
        >>> generate_unique_id("chunk")
        'chunk_a1b2c3d4-e5f6-...'
    """
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def compute_checksum(text: str, algorithm: str = "sha256") -> str:
    """
    Compute cryptographic checksum of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (sha256, md5)
    
    Returns:
        Checksum string in format '<algorithm>:<hex_digest>'
    
    Security note:
        - Use SHA256 for security-sensitive operations
        - MD5 acceptable only for non-security deduplication
    
    Example:
        >>> compute_checksum("hello")
        'sha256:2cf24dba5fb0a3...'
    """
    if algorithm == "sha256":
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    elif algorithm == "md5":
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return f"{algorithm}:{digest}"


class Timer:
    """
    Simple context manager for timing operations.
    
    Example:
        >>> with Timer("embedding"):
        ...     # expensive operation
        ...     pass
        [INFO] embedding completed in 1.23s
    """
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} completed in {self.elapsed:.2f}s")


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp in ISO 8601 format.
    
    Args:
        dt: Datetime object (defaults to now)
    
    Returns:
        ISO formatted timestamp string
    
    Example:
        >>> format_timestamp()
        '2026-01-10T12:34:56.789Z'
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + "Z"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to append when truncated
    
    Returns:
        Truncated text
    
    Example:
        >>> truncate_text("A very long sentence", 10)
        'A very...'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


# TODO: Add utilities as needed:
# - retry decorator with exponential backoff
# - rate limiter for API calls
# - batch processing helpers
# - file I/O helpers (read_text_file, write_json, etc.)
