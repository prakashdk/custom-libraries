"""
RAG Service Module.

This module provides a high-level RAGService class for Retrieval-Augmented Generation.
Framework-agnostic service with sensible defaults that can be used in CLI apps,
web servers, notebooks, or any Python application.

Classes:
    RAGService: High-level RAG service with lifecycle management

Example:
    >>> from llama_rag import RAGService
    >>> from pathlib import Path
    >>> 
    >>> service = RAGService()
    >>> service.ingest_from_directory(Path("./docs"))
    >>> answer = service.query("What is this about?")
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile

from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from llama_rag.rag import SimpleRAG
from llama_rag.utils import get_logger

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
        
        # Create Document objects with metadata
        from langchain_core.documents import Document
        
        documents = []
        for idx, text in enumerate(texts):
            doc_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
            # Add index as fallback metadata
            if not doc_metadata:
                doc_metadata = {"index": idx}
            documents.append(Document(page_content=text, metadata=doc_metadata))
        
        # Ingest using SimpleRAG with documents directly
        self.rag.ingest_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
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
    
    def free_query(self, context: str, question: str) -> str:
        """
        Query LLM directly with provided context (no retrieval).
        
        Useful for:
        - Custom context that's not in the index
        - Testing prompts without retrieval
        - One-off questions with specific context
        
        Args:
            context: Context text to provide to the LLM
            question: Question to ask
        
        Returns:
            Generated response
        
        Example:
            >>> context = "Python is a programming language."
            >>> answer = service.free_query(context, "What is Python?")
        """
        logger.info(f"Free query: {question[:100]}")
        
        try:
            # Use the RAG's chain directly with provided context
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            template = """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.rag.llm | StrOutputParser()
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return response
            
        except Exception as e:
            if "Connection refused" in str(e) or "ConnectionError" in str(type(e).__name__):
                raise RuntimeError(
                    f"Cannot connect to Ollama. Please ensure Ollama is running:\n"
                    f"  ollama serve\n"
                    f"Original error: {e}"
                )
            raise
    
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
    
    def reindex_all(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Reindex all documents in the current index.
        
        Useful for:
        - Changing chunk size or overlap settings
        - Updating embeddings with a different model
        - Rebuilding corrupted index
        
        This will:
        1. Extract all documents from current index
        2. Clear the index
        3. Re-chunk and re-embed all documents
        
        Args:
            chunk_size: New chunk size (defaults to current service config)
            chunk_overlap: New chunk overlap (defaults to current service config)
        
        Returns:
            Number of documents reindexed
        
        Raises:
            ValueError: If no index is loaded
        
        Example:
            >>> # Reindex with different chunk size
            >>> service.reindex_all(chunk_size=1000, chunk_overlap=100)
        """
        if not self.has_index():
            raise ValueError("No index available. Cannot reindex without existing documents.")
        
        logger.info("Starting reindex of all documents")
        
        # Get current documents from vectorstore
        # Note: This extracts the raw text from all chunks
        try:
            # Get all documents from vectorstore
            # FAISS doesn't have a direct way to get all docs, so we use docstore
            if hasattr(self.rag.vectorstore, 'docstore'):
                all_docs = list(self.rag.vectorstore.docstore._dict.values())
                logger.info(f"Found {len(all_docs)} chunks to reindex")
            else:
                raise ValueError("Vectorstore doesn't support document extraction")
            
            if not all_docs:
                logger.warning("No documents found in index")
                return 0
            
            # Extract unique source texts (combine chunks from same source)
            source_texts = {}
            for doc in all_docs:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_texts:
                    source_texts[source] = []
                source_texts[source].append(doc.page_content)
            
            # Combine chunks from same source
            texts = []
            metadata = []
            for source, chunks in source_texts.items():
                # Join chunks with newlines
                combined_text = "\n\n".join(chunks)
                texts.append(combined_text)
                metadata.append({'source': source})
            
            logger.info(f"Extracted {len(texts)} unique documents")
            
            # Clear current index
            self.rag.vectorstore = None
            logger.info("Cleared existing index")
            
            # Use new chunk settings or current ones
            chunk_size = chunk_size or self.chunk_size
            chunk_overlap = chunk_overlap or self.chunk_overlap
            
            # Update service config
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            
            # Reingest with new settings
            count = self.ingest_from_texts(
                texts=texts,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            logger.info(f"Successfully reindexed {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"Failed to reindex: {e}")
            raise RuntimeError(f"Reindex failed: {e}. Index may be in inconsistent state.")
