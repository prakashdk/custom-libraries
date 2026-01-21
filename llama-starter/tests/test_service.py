"""
tests/test_service.py

Comprehensive tests for the service layer (knowledge + records modes).
"""

from pathlib import Path
import pytest

from llama_rag.service import KnowledgeBaseService, RecordsService


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
    """Create a temporary knowledge-base index directory."""
    index_dir = tmp_path / "knowledge_index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def temp_records_index(tmp_path):
    """Create a temporary records index directory."""
    index_dir = tmp_path / "records_index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def knowledge_service(temp_index):
    """Create a KnowledgeBaseService instance."""
    return KnowledgeBaseService(
        index_path=temp_index,
        auto_load=False,
        auto_save=False,
    )


@pytest.fixture
def records_service(temp_records_index):
    """Create a RecordsService instance."""
    return RecordsService(
        index_path=temp_records_index,
        auto_load=False,
        auto_save=False,
    )


class TestKnowledgeBaseServiceInit:
    """Test KnowledgeBaseService initialization."""
    
    def test_default_initialization(self, temp_index):
        """Test service initializes with defaults."""
        service = KnowledgeBaseService(index_path=temp_index, auto_load=False)
        
        assert service.index_path == temp_index
        assert service.embedding_type == "ollama"
        assert service.embedding_model == "embeddinggemma"
        assert service.llm_type == "ollama"
        assert service.llm_model == "llama3.2"
        assert service.chunk_size == 500
        assert service.retrieval_k == 4
    
    def test_custom_initialization(self, temp_index):
        """Test service initializes with custom config."""
        service = KnowledgeBaseService(
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
    
    def test_has_index_false_initially(self, knowledge_service):
        """Test has_index returns False for new service."""
        assert knowledge_service.has_index() is False


class TestKnowledgeBaseServiceIngestion:
    """Test document ingestion methods."""
    
    def test_ingest_from_directory(self, knowledge_service, temp_corpus):
        """Test ingesting documents from directory."""
        count = knowledge_service.ingest_from_directory(temp_corpus)
        
        assert count > 0
        assert knowledge_service.has_index() is True
    
    def test_ingest_from_nonexistent_directory(self, knowledge_service):
        """Test error when ingesting from nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            knowledge_service.ingest_from_directory(Path("/nonexistent/path"))
    
    def test_ingest_from_texts(self, knowledge_service):
        """Test ingesting from text strings."""
        texts = [
            "This is document one about machine learning.",
            "This is document two about deep learning.",
            "This is document three about neural networks.",
        ]
        
        count = knowledge_service.ingest_from_texts(texts)
        
        assert count == 3
        assert knowledge_service.has_index() is True
    
    def test_ingest_from_texts_with_metadata(self, knowledge_service):
        """Test ingesting texts with metadata."""
        texts = ["Document one", "Document two"]
        metadata = [
            {"source": "doc1", "author": "Alice"},
            {"source": "doc2", "author": "Bob"},
        ]
        
        count = knowledge_service.ingest_from_texts(texts, metadata=metadata)
        
        assert count == 2
        assert knowledge_service.has_index() is True
    
    def test_add_document(self, knowledge_service):
        """Test adding a single document."""
        count = knowledge_service.add_document(
            "This is a single document about AI.",
            metadata={"source": "manual", "type": "note"}
        )
        
        assert count == 1
        assert knowledge_service.has_index() is True
    
    def test_custom_chunking(self, knowledge_service, temp_corpus):
        """Test ingestion with custom chunk size."""
        count = knowledge_service.ingest_from_directory(
            temp_corpus,
            chunk_size=100,
            chunk_overlap=10
        )
        
        assert count > 0
        assert knowledge_service.has_index() is True


class TestKnowledgeBaseServiceQuery:
    """Test query and retrieve methods for knowledge mode."""
    
    def test_query_without_index(self, knowledge_service):
        """Test error when querying without index."""
        with pytest.raises(ValueError, match="No index available"):
            knowledge_service.query("What is machine learning?")
    
    def test_retrieve_without_index(self, knowledge_service):
        """Test error when retrieving without index."""
        with pytest.raises(ValueError, match="No index available"):
            knowledge_service.retrieve("What is machine learning?")
    
    @pytest.mark.integration
    def test_query_after_ingestion(self, knowledge_service, temp_corpus):
        """Test querying after ingesting documents."""
        knowledge_service.ingest_from_directory(temp_corpus)
        
        response = knowledge_service.query("What is machine learning?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.integration
    def test_retrieve_after_ingestion(self, knowledge_service, temp_corpus):
        """Test retrieving documents after ingestion."""
        knowledge_service.ingest_from_directory(temp_corpus)
        
        docs = knowledge_service.retrieve("machine learning", k=2)
        
        assert len(docs) <= 2
        assert all(hasattr(doc, 'page_content') for doc in docs)
    
    @pytest.mark.integration
    def test_query_with_custom_k(self, knowledge_service, temp_corpus):
        """Test querying with custom retrieval count."""
        knowledge_service.ingest_from_directory(temp_corpus)
        
        response = knowledge_service.query("What is AI?", k=1)
        
        assert isinstance(response, str)


class TestKnowledgeBaseServicePersistence:
    """Test save and load functionality."""
    
    def test_save_empty_index(self, knowledge_service, temp_index):
        """Test saving when no index exists."""
        knowledge_service.save()  # Should not raise, just log warning
    
    @pytest.mark.integration
    def test_save_and_load(self, temp_corpus, temp_index):
        """Test saving and loading index."""
        # Create and save
        service1 = KnowledgeBaseService(index_path=temp_index, auto_load=False, auto_save=False)
        service1.ingest_from_directory(temp_corpus)
        service1.save()
        
        assert temp_index.exists()
        assert list(temp_index.glob("*"))  # Has files
        
        # Load in new service
        service2 = KnowledgeBaseService(index_path=temp_index, auto_load=True, auto_save=False)
        
        assert service2.has_index() is True
    
    @pytest.mark.integration
    def test_auto_save(self, temp_corpus, temp_index):
        """Test auto-save functionality."""
        service = KnowledgeBaseService(index_path=temp_index, auto_load=False, auto_save=True)
        service.ingest_from_directory(temp_corpus)
        
        # Should auto-save
        assert temp_index.exists()
        assert list(temp_index.glob("*"))
    
    @pytest.mark.integration
    def test_auto_load(self, temp_corpus, temp_index):
        """Test auto-load functionality."""
        # Create and save
        service1 = KnowledgeBaseService(index_path=temp_index, auto_load=False, auto_save=False)
        service1.ingest_from_directory(temp_corpus)
        service1.save()
        
        # Auto-load in new service
        service2 = KnowledgeBaseService(index_path=temp_index, auto_load=True, auto_save=False)
        
        assert service2.has_index() is True


class TestKnowledgeBaseServiceContextManager:
    """Test context manager functionality."""
    
    @pytest.mark.integration
    def test_context_manager_auto_save(self, temp_corpus, temp_index):
        """Test context manager auto-saves on exit."""
        with KnowledgeBaseService(index_path=temp_index, auto_load=False, auto_save=True) as service:
            service.ingest_from_directory(temp_corpus)
        
        # Should be saved after context exit
        assert temp_index.exists()
        assert list(temp_index.glob("*"))
    
    @pytest.mark.integration
    def test_context_manager_no_auto_save(self, temp_corpus, temp_index):
        """Test context manager respects auto_save=False."""
        with KnowledgeBaseService(index_path=temp_index, auto_load=False, auto_save=False) as service:
            service.ingest_from_directory(temp_corpus)
        
        # Should not save automatically
        assert not list(temp_index.glob("*"))


class TestKnowledgeBaseServiceUtilities:
    """Test utility methods for knowledge mode."""
    
    def test_get_stats_empty(self, knowledge_service):
        """Test get_stats on empty service."""
        stats = knowledge_service.get_stats()
        
        assert stats["has_index"] is False
        assert stats["embedding_type"] == "ollama"
        assert stats["llm_type"] == "ollama"
        assert stats["chunk_size"] == 500
        assert stats["retrieval_k"] == 4
    
    @pytest.mark.integration
    def test_get_stats_after_ingestion(self, knowledge_service, temp_corpus):
        """Test get_stats after ingesting documents."""
        knowledge_service.ingest_from_directory(temp_corpus)
        
        stats = knowledge_service.get_stats()
        
        assert stats["has_index"] is True
        assert stats["index_path"] == str(knowledge_service.index_path)
    
    def test_reset(self, knowledge_service, temp_corpus):
        """Test resetting the service."""
        knowledge_service.ingest_from_directory(temp_corpus)
        assert knowledge_service.has_index() is True
        
        knowledge_service.reset()
        assert knowledge_service.has_index() is False


class TestRecordsService:
    """Tests specific to the RecordsService mode."""

    def test_search_without_index(self, records_service):
        """Searching without an index should error."""
        with pytest.raises(ValueError, match="No index available"):
            records_service.search("test")

    def test_add_record_alias(self, records_service):
        """add_record should ingest data and expose metadata."""
        count = records_service.add_record(
            "Records mode document",
            metadata={"category": "records", "id": 42}
        )

        assert count == 1
        assert records_service.has_index() is True

    @pytest.mark.integration
    def test_search_after_ingestion(self, records_service, temp_corpus):
        """Search should return chunks once documents are ingested."""
        records_service.ingest_from_directory(temp_corpus)
        docs = records_service.search("machine learning", k=2)

        assert len(docs) <= 2
        assert all(hasattr(doc, "metadata") for doc in docs)


class TestServiceIsolation:
    """Ensure default index paths stay isolated per mode."""

    def test_default_index_paths_do_not_collide(self):
        kb_service = KnowledgeBaseService(auto_load=False, auto_save=False)
        records = RecordsService(auto_load=False, auto_save=False)

        assert kb_service.index_path != records.index_path
        assert kb_service.index_path.name == "knowledge_index"
        assert records.index_path.name == "records_index"
