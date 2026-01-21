"""
config/env.py

Environment variable loading and configuration helpers.

This module is deprecated - use common/config.py instead.
Kept for backward compatibility.
"""

from common.config import get_config, get_config_value

__all__ = ["get_config", "get_config_value"]
