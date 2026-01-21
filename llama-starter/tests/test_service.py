"""
tests/test_service.py

Comprehensive tests for RAGService.
"""

import tempfile
from pathlib import Path
import pytest

from llama_rag.service import RAGService


@pytest.fixture
def temp_corpus(tmp_path):
    """Create a temporary corpus with sample documents."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    
    # Create sample documents
    (corpus_dir / "doc1.txt").write_text("Machine learning is a subset of artificial intelligence.")
    (corpus_dir / "doc2.txt").write_text("Deep learning uses neural networks with multiple layers.")
    (corpus_dir / "doc3.md").write_text("# NLP\nNatural language processing deals with text and speech.")
    
    return corpus_dir


@pytest.fixture
def temp_index(tmp_path):
    """Create a temporary index directory."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def service(temp_index):
    """Create a RAGService instance."""
    return RAGService(
        index_path=temp_index,
        auto_load=False,
        auto_save=False,
    )


class TestRAGServiceInit:
    """Test RAGService initialization."""
    
    def test_default_initialization(self, temp_index):
        """Test service initializes with defaults."""
        service = RAGService(index_path=temp_index, auto_load=False)
        
        assert service.index_path == temp_index
        assert service.embedding_type == "ollama"
        assert service.embedding_model == "embeddinggemma"
        assert service.llm_type == "ollama"
        assert service.llm_model == "llama3.1"
        assert service.chunk_size == 500
        assert service.retrieval_k == 4
    
    def test_custom_initialization(self, temp_index):
        """Test service initializes with custom config."""
        service = RAGService(
            index_path=temp_index,
            auto_load=False,
            embedding_model="custom-model",
            llm_model="custom-llm",
            chunk_size=1000,
            retrieval_k=10,
        )
        
        assert service.embedding_model == "custom-model"
        assert service.llm_model == "custom-llm"
        assert service.chunk_size == 1000
        assert service.retrieval_k == 10
    
    def test_has_index_false_initially(self, service):
        """Test has_index returns False for new service."""
        assert service.has_index() is False


class TestRAGServiceIngestion:
    """Test document ingestion methods."""
    
    def test_ingest_from_directory(self, service, temp_corpus):
        """Test ingesting documents from directory."""
        count = service.ingest_from_directory(temp_corpus)
        
        assert count > 0
        assert service.has_index() is True
    
    def test_ingest_from_nonexistent_directory(self, service):
        """Test error when ingesting from nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            service.ingest_from_directory(Path("/nonexistent/path"))
    
    def test_ingest_from_texts(self, service):
        """Test ingesting from text strings."""
        texts = [
            "This is document one about machine learning.",
            "This is document two about deep learning.",
            "This is document three about neural networks.",
        ]
        
        count = service.ingest_from_texts(texts)
        
        assert count == 3
        assert service.has_index() is True
    
    def test_ingest_from_texts_with_metadata(self, service):
        """Test ingesting texts with metadata."""
        texts = ["Document one", "Document two"]
        metadata = [
            {"source": "doc1", "author": "Alice"},
            {"source": "doc2", "author": "Bob"},
        ]
        
        count = service.ingest_from_texts(texts, metadata=metadata)
        
        assert count == 2
        assert service.has_index() is True
    
    def test_add_document(self, service):
        """Test adding a single document."""
        count = service.add_document(
            "This is a single document about AI.",
            metadata={"source": "manual", "type": "note"}
        )
        
        assert count == 1
        assert service.has_index() is True
    
    def test_custom_chunking(self, service, temp_corpus):
        """Test ingestion with custom chunk size."""
        count = service.ingest_from_directory(
            temp_corpus,
            chunk_size=100,
            chunk_overlap=10
        )
        
        assert count > 0
        assert service.has_index() is True


class TestRAGServiceQuery:
    """Test query and retrieve methods."""
    
    def test_query_without_index(self, service):
        """Test error when querying without index."""
        with pytest.raises(ValueError, match="No index available"):
            service.query("What is machine learning?")
    
    def test_retrieve_without_index(self, service):
        """Test error when retrieving without index."""
        with pytest.raises(ValueError, match="No index available"):
            service.retrieve("What is machine learning?")
    
    @pytest.mark.integration
    def test_query_after_ingestion(self, service, temp_corpus):
        """Test querying after ingesting documents."""
        service.ingest_from_directory(temp_corpus)
        
        response = service.query("What is machine learning?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.integration
    def test_retrieve_after_ingestion(self, service, temp_corpus):
        """Test retrieving documents after ingestion."""
        service.ingest_from_directory(temp_corpus)
        
        docs = service.retrieve("machine learning", k=2)
        
        assert len(docs) <= 2
        assert all(hasattr(doc, 'page_content') for doc in docs)
    
    @pytest.mark.integration
    def test_query_with_custom_k(self, service, temp_corpus):
        """Test querying with custom retrieval count."""
        service.ingest_from_directory(temp_corpus)
        
        response = service.query("What is AI?", k=1)
        
        assert isinstance(response, str)


class TestRAGServicePersistence:
    """Test save and load functionality."""
    
    def test_save_empty_index(self, service, temp_index):
        """Test saving when no index exists."""
        service.save()  # Should not raise, just log warning
    
    @pytest.mark.integration
    def test_save_and_load(self, temp_corpus, temp_index):
        """Test saving and loading index."""
        # Create and save
        service1 = RAGService(index_path=temp_index, auto_load=False, auto_save=False)
        service1.ingest_from_directory(temp_corpus)
        service1.save()
        
        assert temp_index.exists()
        assert list(temp_index.glob("*"))  # Has files
        
        # Load in new service
        service2 = RAGService(index_path=temp_index, auto_load=True, auto_save=False)
        
        assert service2.has_index() is True
    
    @pytest.mark.integration
    def test_auto_save(self, temp_corpus, temp_index):
        """Test auto-save functionality."""
        service = RAGService(index_path=temp_index, auto_load=False, auto_save=True)
        service.ingest_from_directory(temp_corpus)
        
        # Should auto-save
        assert temp_index.exists()
        assert list(temp_index.glob("*"))
    
    @pytest.mark.integration
    def test_auto_load(self, temp_corpus, temp_index):
        """Test auto-load functionality."""
        # Create and save
        service1 = RAGService(index_path=temp_index, auto_load=False, auto_save=False)
        service1.ingest_from_directory(temp_corpus)
        service1.save()
        
        # Auto-load in new service
        service2 = RAGService(index_path=temp_index, auto_load=True, auto_save=False)
        
        assert service2.has_index() is True


class TestRAGServiceContextManager:
    """Test context manager functionality."""
    
    @pytest.mark.integration
    def test_context_manager_auto_save(self, temp_corpus, temp_index):
        """Test context manager auto-saves on exit."""
        with RAGService(index_path=temp_index, auto_load=False, auto_save=True) as service:
            service.ingest_from_directory(temp_corpus)
        
        # Should be saved after context exit
        assert temp_index.exists()
        assert list(temp_index.glob("*"))
    
    @pytest.mark.integration
    def test_context_manager_no_auto_save(self, temp_corpus, temp_index):
        """Test context manager respects auto_save=False."""
        with RAGService(index_path=temp_index, auto_load=False, auto_save=False) as service:
            service.ingest_from_directory(temp_corpus)
        
        # Should not save automatically
        assert not list(temp_index.glob("*"))


class TestRAGServiceUtilities:
    """Test utility methods."""
    
    def test_get_stats_empty(self, service):
        """Test get_stats on empty service."""
        stats = service.get_stats()
        
        assert stats["has_index"] is False
        assert stats["embedding_type"] == "ollama"
        assert stats["llm_type"] == "ollama"
        assert stats["chunk_size"] == 500
        assert stats["retrieval_k"] == 4
    
    @pytest.mark.integration
    def test_get_stats_after_ingestion(self, service, temp_corpus):
        """Test get_stats after ingesting documents."""
        service.ingest_from_directory(temp_corpus)
        
        stats = service.get_stats()
        
        assert stats["has_index"] is True
        assert stats["index_path"] == str(service.index_path)
    
    def test_reset(self, service, temp_corpus):
        """Test resetting the service."""
        service.ingest_from_directory(temp_corpus)
        assert service.has_index() is True
        
        service.reset()
        assert service.has_index() is False
