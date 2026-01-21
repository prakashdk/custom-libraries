"""
common/service.py

Reusable RAG service layer.
Framework-agnostic service with sensible defaults.
Config is passed from entry points, not read directly.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile

from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from common.rag import SimpleRAG
from common.utils import get_logger

logger = get_logger(__name__)
console = Console()


class RAGService:
    """
    High-level RAG service with lifecycle management and sensible defaults.
    
    This service is framework-agnostic and can be used by:
    - CLI applications
    - FastAPI/Flask servers
    - Jupyter notebooks
    - Any other Python code
    
    Features:
    - Sensible defaults (Ollama + FAISS)
    - Auto-load/auto-save index
    - Multiple ingestion sources
    - Query with/without generation
    """
    
    # Service-level defaults
    DEFAULT_EMBEDDING_TYPE = "ollama"
    DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
    DEFAULT_LLM_TYPE = "ollama"
    DEFAULT_LLM_MODEL = "llama3.2"
    DEFAULT_LLM_TEMPERATURE = 0.7
    DEFAULT_VECTORSTORE_TYPE = "faiss"
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_RETRIEVAL_K = 4
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        auto_load: bool = True,
        auto_save: bool = True,
        # Model configuration
        embedding_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_type: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        vectorstore_type: Optional[str] = None,
        # Processing configuration
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        retrieval_k: Optional[int] = None,
    ):
        """
        Initialize RAG service.
        
        Args:
            index_path: Path to index directory
            auto_load: Whether to auto-load existing index on init
            auto_save: Whether to auto-save after ingestion
            embedding_type: Embedding type (defaults to ollama)
            embedding_model: Embedding model (defaults to embeddinggemma)
            llm_type: LLM type (defaults to ollama)
            llm_model: LLM model (defaults to llama3.2)
            llm_temperature: LLM temperature (defaults to 0.7)
            vectorstore_type: Vector store type (defaults to faiss)
            chunk_size: Chunk size (defaults to 500)
            chunk_overlap: Chunk overlap (defaults to 50)
            retrieval_k: Number of docs to retrieve (defaults to 4)
        """
        self.index_path = index_path or Path("./data/index")
        self.auto_save = auto_save
        
        # Store configuration with defaults
        self.embedding_type = embedding_type or self.DEFAULT_EMBEDDING_TYPE
        self.embedding_model = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.llm_type = llm_type or self.DEFAULT_LLM_TYPE
        self.llm_model = llm_model or self.DEFAULT_LLM_MODEL
        self.llm_temperature = llm_temperature if llm_temperature is not None else self.DEFAULT_LLM_TEMPERATURE
        self.vectorstore_type = vectorstore_type or self.DEFAULT_VECTORSTORE_TYPE
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        self.retrieval_k = retrieval_k or self.DEFAULT_RETRIEVAL_K
        
        # Create SimpleRAG with config
        self.rag = SimpleRAG(
            embedding_type=self.embedding_type,
            embedding_model=self.embedding_model,
            llm_type=self.llm_type,
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
            vectorstore_type=self.vectorstore_type,
        )
        
        # Auto-load existing index
        if auto_load and self.index_path.exists():
            try:
                self.rag.load(self.index_path)
                logger.info(f"Loaded existing index from: {self.index_path}")
            except Exception as e:
                logger.warning(f"Could not load index from {self.index_path}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-save if configured."""
        if self.auto_save and self.has_index():
            try:
                self.save()
                logger.info("Auto-saved index on context exit")
            except Exception as e:
                logger.error(f"Failed to auto-save on exit: {e}")
        return False
    
    def ingest_from_directory(
        self,
        directory: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Ingest documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            chunk_size: Chunk size (defaults to service config)
            chunk_overlap: Chunk overlap (defaults to service config)
        
        Returns:
            Number of documents ingested
        
        Raises:
            FileNotFoundError: If directory doesn't exist
            RuntimeError: If Ollama is not running (when using Ollama)
        """
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}\n"
                f"Please check the path and try again."
            )
        
        logger.info(f"Ingesting from directory: {directory}")
        
        # Use service defaults if not specified
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        # Show progress in non-verbose mode
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True,  # Remove progress bar when done
            ) as progress:
                task = progress.add_task("Processing documents...", total=None)
                
                # Ingest
                self.rag.ingest(directory, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                progress.update(task, completed=True)
        except Exception as e:
            # Better error messages for common issues
            if "Connection refused" in str(e) or "ConnectionError" in str(type(e).__name__):
                raise RuntimeError(
                    f"Cannot connect to Ollama. Please ensure Ollama is running:\n"
                    f"  1. Start Ollama: ollama serve\n"
                    f"  2. Pull models: ollama pull embeddinggemma && ollama pull llama3.1\n"
                    f"Original error: {e}"
                )
            raise
        
        # Auto-save
        if self.auto_save:
            self.save()
        
        logger.info(f"Successfully ingested from: {directory}")
        return len(list(directory.glob("*")))
    
    def ingest_from_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Ingest documents from raw text strings.
        
        Args:
            texts: List of text documents
            metadata: Optional metadata per document
            chunk_size: Chunk size (defaults to service config)
            chunk_overlap: Chunk overlap (defaults to service config)
        
        Returns:
            Number of documents ingested
        """
        logger.info(f"Ingesting {len(texts)} text documents")
        
        # Use service defaults if not specified
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        # Create temporary directory with text files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write texts to files
            for idx, text in enumerate(texts):
                file_path = tmppath / f"doc_{idx}.txt"
                file_path.write_text(text)
            
            # Ingest using SimpleRAG
            self.rag.ingest(tmppath, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Auto-save
        if self.auto_save:
            self.save()
        
        logger.info(f"Successfully ingested {len(texts)} documents")
        return len(texts)
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Add a single document directly to the index.
        
        Args:
            text: Document text
            metadata: Optional metadata for the document
            chunk_size: Chunk size (defaults to service config)
            chunk_overlap: Chunk overlap (defaults to service config)
        
        Returns:
            Number of chunks added (1 document may create multiple chunks)
        """
        logger.info(f"Adding single document")
        
        # Use ingest_from_texts with single document
        return self.ingest_from_texts(
            texts=[text],
            metadata=[metadata] if metadata else None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def query(self, query: str, k: Optional[int] = None) -> str:
        """
        Query with RAG generation.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (defaults to service config)
        
        Returns:
            Generated response
        
        Raises:
            ValueError: If no index is loaded
            RuntimeError: If Ollama is not running (when using Ollama)
        """
        if not self.has_index():
            raise ValueError(
                f"No index available. Please ingest documents first:\n"
                f"  python -m app.cli ingest <directory> --output {self.index_path}\n"
                f"Or load an existing index:\n"
                f"  service.load(Path('{self.index_path}'))"
            )
        
        k = k or self.retrieval_k
        
        logger.info(f"Querying: {query[:100]}")
        
        try:
            response = self.rag.query(query, k=k)
        except Exception as e:
            if "Connection refused" in str(e) or "ConnectionError" in str(type(e).__name__):
                raise RuntimeError(
                    f"Cannot connect to Ollama. Please ensure Ollama is running:\n"
                    f"  ollama serve\n"
                    f"Original error: {e}"
                )
            raise
        
        return response
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve documents without generation.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (defaults to service config)
        
        Returns:
            List of retrieved documents
        """
        if not self.has_index():
            raise ValueError("No index available. Ingest documents first.")
        
        k = k or self.retrieval_k
        
        logger.info(f"Retrieving for: {query[:100]}")
        docs = self.rag.retrieve(query, k=k)
        
        return docs
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save index to disk.
        
        Args:
            path: Path to save to (defaults to configured index_path)
        """
        save_path = path or self.index_path
        
        if not self.has_index():
            logger.warning("No index to save")
            return
        
        self.rag.save(save_path)
        logger.info(f"Saved index to: {save_path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """
        Load index from disk.
        
        Args:
            path: Path to load from (defaults to configured index_path)
        """
        load_path = path or self.index_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index not found: {load_path}")
        
        self.rag.load(load_path)
        logger.info(f"Loaded index from: {load_path}")
    
    def has_index(self) -> bool:
        """Check if index is loaded."""
        return self.rag.vectorstore is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dictionary with service stats
        """
        return {
            "has_index": self.has_index(),
            "index_path": str(self.index_path),
            "embedding_type": self.embedding_type,
            "embedding_model": self.embedding_model,
            "llm_type": self.llm_type,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "vectorstore_type": self.vectorstore_type,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "auto_save": self.auto_save,
        }
    
    def reset(self) -> None:
        """Reset service (clear index)."""
        self.rag = SimpleRAG()
        logger.info("Service reset")
