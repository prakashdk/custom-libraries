"""
tests/integration_test_rag.py

Integration test for end-to-end RAG workflow using LangChain.

Tests the complete flow:
1. Ingest documents
2. Query with RAG
3. Retrieve documents
4. Save/load index
"""

import pytest
import tempfile
from pathlib import Path

from llama_rag.rag import SimpleRAG, load_documents, split_documents


@pytest.fixture
def sample_corpus(tmp_path):
    """Create sample corpus for testing."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    
    # Create sample documents
    (corpus_dir / "ml.txt").write_text(
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without being explicitly programmed."
    )
    
    (corpus_dir / "dl.txt").write_text(
        "Deep learning uses neural networks with multiple layers. "
        "It has revolutionized computer vision and natural language processing."
    )
    
    (corpus_dir / "nlp.txt").write_text(
        "Natural language processing helps computers understand human language. "
        "It includes tasks like translation, sentiment analysis, and question answering."
    )
    
    return corpus_dir


@pytest.fixture
def index_dir(tmp_path):
    """Create temporary index directory."""
    idx_dir = tmp_path / "index"
    idx_dir.mkdir()
    return idx_dir


def test_load_documents(sample_corpus):
    """Test document loading."""
    docs = load_documents(sample_corpus)
    
    assert len(docs) == 3
    assert all(doc.page_content for doc in docs)
    assert all(doc.metadata.get("source") for doc in docs)


def test_split_documents(sample_corpus):
    """Test document splitting."""
    docs = load_documents(sample_corpus)
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)
    
    # Should create more chunks than original docs
    assert len(chunks) >= len(docs)
    
    # Each chunk should have content
    assert all(chunk.page_content for chunk in chunks)


def test_simple_rag_ingest(sample_corpus):
    """Test RAG ingestion."""
    rag = SimpleRAG()
    rag.ingest(sample_corpus, chunk_size=100, chunk_overlap=20)
    
    # Verify vectorstore was created
    assert rag.vectorstore is not None
    assert rag.retriever is not None
    assert rag.chain is not None


def test_simple_rag_retrieve(sample_corpus):
    """Test document retrieval."""
    rag = SimpleRAG()
    rag.ingest(sample_corpus)
    
    # Retrieve documents
    docs = rag.retrieve("What is machine learning?", k=2)
    
    assert len(docs) <= 2
    assert all(doc.page_content for doc in docs)
    
    # The first result should be relevant to ML
    assert "machine learning" in docs[0].page_content.lower() or "artificial intelligence" in docs[0].page_content.lower()


def test_simple_rag_query(sample_corpus):
    """Test RAG query with generation."""
    rag = SimpleRAG()
    rag.ingest(sample_corpus)
    
    # Query with generation
    response = rag.query("What is deep learning?", k=2)
    
    assert isinstance(response, str)
    assert len(response) > 0


def test_simple_rag_save_load(sample_corpus, index_dir):
    """Test saving and loading index."""
    # Create and save index
    rag1 = SimpleRAG()
    rag1.ingest(sample_corpus)
    rag1.save(index_dir)
    
    # Verify files were created
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()
    
    # Load index
    rag2 = SimpleRAG()
    rag2.load(index_dir)
    
    # Verify it works
    docs = rag2.retrieve("What is NLP?", k=2)
    assert len(docs) > 0
    assert any("natural language" in doc.page_content.lower() for doc in docs)


def test_simple_rag_empty_query(sample_corpus):
    """Test error handling for empty queries."""
    rag = SimpleRAG()
    rag.ingest(sample_corpus)
    
    with pytest.raises(Exception):
        rag.query("")


def test_simple_rag_no_index():
    """Test error handling when querying without index."""
    rag = SimpleRAG()
    
    with pytest.raises(ValueError):
        rag.query("test query")


def test_simple_rag_custom_config(sample_corpus):
    """Test RAG with custom configuration."""
    rag = SimpleRAG()
    
    # Ingest with custom parameters
    rag.ingest(
        sample_corpus,
        chunk_size=50,
        chunk_overlap=5
    )
    
    # Query with custom k
    docs = rag.retrieve("machine learning", k=1)
    assert len(docs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
