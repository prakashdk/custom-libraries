"""
tests/test_factories.py

Unit tests for factory functions.
"""

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from llama_rag.factories import get_embeddings, get_llm, get_vectorstore


def test_get_embeddings_ollama():
    """Test Ollama embeddings factory."""
    embeddings = get_embeddings("ollama", "embeddinggemma")
    
    assert isinstance(embeddings, Embeddings)
    assert embeddings.model == "embeddinggemma"


def test_get_embeddings_default():
    """Test default embeddings factory."""
    embeddings = get_embeddings()
    
    assert isinstance(embeddings, Embeddings)


def test_get_embeddings_invalid():
    """Test invalid embeddings type."""
    with pytest.raises(ValueError):
        get_embeddings("invalid_type")


def test_get_llm_ollama():
    """Test Ollama LLM factory."""
    llm = get_llm("ollama", "llama3.2")
    
    assert isinstance(llm, BaseChatModel)


def test_get_llm_default():
    """Test default LLM factory."""
    llm = get_llm()
    
    assert isinstance(llm, BaseChatModel)


def test_get_llm_with_temperature():
    """Test LLM factory with temperature."""
    llm = get_llm("ollama", "llama3.2", temperature=0.5)
    
    assert isinstance(llm, BaseChatModel)
    assert llm.temperature == 0.5


def test_get_llm_invalid():
    """Test invalid LLM type."""
    with pytest.raises(ValueError):
        get_llm("invalid_type")


def test_get_vectorstore_faiss():
    """Test FAISS vectorstore factory."""
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings, "faiss")
    
    # FAISS needs documents, so this returns a class
    assert vectorstore is not None


def test_get_vectorstore_default():
    """Test default vectorstore factory."""
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    
    assert vectorstore is not None


def test_get_vectorstore_invalid():
    """Test invalid vectorstore type."""
    embeddings = get_embeddings()
    
    with pytest.raises(ValueError):
        get_vectorstore(embeddings, "invalid_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
