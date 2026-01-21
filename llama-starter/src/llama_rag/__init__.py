"""
llama-rag-lib: A simple RAG library using LangChain and Ollama

This library provides a high-level RAG (Retrieval-Augmented Generation) service
that can be easily integrated into any Python application.

Main exports:
    - RAGService: High-level service for RAG operations
    - SimpleRAG: Core RAG implementation
    - Factory functions for creating LangChain components
    - Document loading and processing utilities

Example:
    >>> from llama_rag import RAGService
    >>> from pathlib import Path
    >>> 
    >>> # Initialize service
    >>> service = RAGService(
    ...     embedding_model="embeddinggemma",
    ...     llm_model="llama3.2",
    ... )
    >>> 
    >>> # Ingest documents
    >>> service.ingest_from_directory(Path("./docs"))
    >>> 
    >>> # Query the system
    >>> answer = service.query("What is RAG?")
    >>> print(answer)
"""

from llama_rag.service import RAGService, KnowledgeBaseService, RecordsService
from llama_rag.rag import SimpleRAG, load_documents, split_documents, create_rag_chain
from llama_rag.factories import get_embeddings, get_llm, get_vectorstore
from llama_rag.config import get_config, load_yaml_config

__version__ = "0.2.0"

__all__ = [
    "RAGService",
    "KnowledgeBaseService",
    "RecordsService",
    "SimpleRAG",
    "load_documents",
    "split_documents",
    "create_rag_chain",
    "get_embeddings",
    "get_llm",
    "get_vectorstore",
    "get_config",
    "load_yaml_config",
]
