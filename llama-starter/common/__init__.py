"""
common/__init__.py

Core library module exports. Simplified using LangChain components.
"""

from common.config import get_config
from common.service import RAGService
from common.rag import SimpleRAG, load_documents, split_documents, create_rag_chain
from common.factories import get_embeddings, get_llm, get_vectorstore

__version__ = "0.1.0"

__all__ = [
    "get_config",
    "RAGService",
    "SimpleRAG",
    "load_documents",
    "split_documents",
    "create_rag_chain",
    "get_embeddings",
    "get_llm",
    "get_vectorstore",
]
