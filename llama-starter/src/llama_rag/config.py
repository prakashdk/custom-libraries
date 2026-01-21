"""
common/config.py

Minimal configuration management for the RAG system.

Most configuration is handled through RAGService constructor parameters.
This module provides optional YAML loading and environment variable overrides.

Loads configuration from:
1. config/defaults.yaml (optional, minimal settings)
2. Environment variables (LLAMA_RAG_* prefix)

Priority: Runtime parameters > Environment > YAML defaults > RAGService defaults
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from llama_rag.utils import get_logger

logger = get_logger(__name__)


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to config/defaults.yaml)
    
    Returns:
        Configuration dictionary (empty if file not found)
    
    Example:
        >>> config = load_yaml_config()
        >>> config.get("index_path")
        './data/index'
    """
    if config_path is None:
        # Default to config/defaults.yaml relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "defaults.yaml"
    
    if not config_path.exists():
        logger.debug(f"Config file not found: {config_path}, using defaults")
        return {}
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config from: {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML config: {e}")
        return {}


def get_env_config() -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables.
    
    All settings use LLAMA_RAG_ prefix.
    
    Supported variables:
        LLAMA_RAG_INDEX_PATH: Path to vector index
        LLAMA_RAG_EMBEDDING_MODEL: Embedding model name
        LLAMA_RAG_LLM_MODEL: LLM model name
        LLAMA_RAG_CHUNK_SIZE: Default chunk size
        LLAMA_RAG_CHUNK_OVERLAP: Default chunk overlap
        LLAMA_RAG_RETRIEVAL_K: Number of documents to retrieve
    
    Returns:
        Dictionary of environment overrides
    
    Example:
        >>> os.environ["LLAMA_RAG_CHUNK_SIZE"] = "1000"
        >>> config = get_env_config()
        >>> config["chunk_size"]
        1000
    """
    env_config: Dict[str, Any] = {}
    
    # Index path
    if index_path := os.getenv("LLAMA_RAG_INDEX_PATH"):
        env_config["index_path"] = index_path
    
    # Embedding model
    if embedding_model := os.getenv("LLAMA_RAG_EMBEDDING_MODEL"):
        env_config["embedding_model"] = embedding_model
    
    # LLM model
    if llm_model := os.getenv("LLAMA_RAG_LLM_MODEL"):
        env_config["llm_model"] = llm_model
    
    # Chunking settings
    if chunk_size := os.getenv("LLAMA_RAG_CHUNK_SIZE"):
        try:
            env_config["chunk_size"] = int(chunk_size)
        except ValueError:
            logger.warning(f"Invalid LLAMA_RAG_CHUNK_SIZE: {chunk_size}")
    
    if chunk_overlap := os.getenv("LLAMA_RAG_CHUNK_OVERLAP"):
        try:
            env_config["chunk_overlap"] = int(chunk_overlap)
        except ValueError:
            logger.warning(f"Invalid LLAMA_RAG_CHUNK_OVERLAP: {chunk_overlap}")
    
    # Retrieval settings
    if retrieval_k := os.getenv("LLAMA_RAG_RETRIEVAL_K"):
        try:
            env_config["retrieval_k"] = int(retrieval_k)
        except ValueError:
            logger.warning(f"Invalid LLAMA_RAG_RETRIEVAL_K: {retrieval_k}")
    
    if env_config:
        logger.debug(f"Environment overrides: {list(env_config.keys())}")
    
    return env_config


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Get merged configuration from YAML and environment variables.
    
    Args:
        reload: Force reload from disk and environment
    
    Returns:
        Merged configuration dictionary
    
    Note:
        Most users should pass parameters directly to RAGService constructor.
        This function is useful for loading shared config across multiple services.
    
    Example:
        >>> config = get_config()
        >>> service = RAGService(
        ...     index_path=config.get("index_path", "./data/index"),
        ...     embedding_model=config.get("embedding_model"),
        ... )
    """
    global _config_cache
    
    if _config_cache is None or reload:
        # Load YAML defaults
        yaml_config = load_yaml_config()
        
        # Load environment overrides
        env_config = get_env_config()
        
        # Merge (env overrides yaml)
        _config_cache = {**yaml_config, **env_config}
        
        logger.debug("Configuration loaded and cached")
    
    return _config_cache.copy()  # Return copy to prevent mutation
