"""
common/rag.py

Simple RAG implementation using LangChain components.
Config is passed from service layer, not read directly.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_rag.factories import get_embeddings, get_llm, get_vectorstore
from llama_rag.utils import get_logger

logger = get_logger(__name__)


def load_documents(
    source_path: str | Path,
) -> List[Document]:
    """
    Load documents from directory or file.
    
    Args:
        source_path: Path to file or directory
    
    Returns:
        List of LangChain Documents
    """
    source_path = Path(source_path)
    
    if source_path.is_file():
        logger.info(f"Loading single file: {source_path}")
        loader = TextLoader(str(source_path))
        return loader.load()
    elif source_path.is_dir():
        logger.info(f"Loading directory: {source_path}")
        
        # Load .txt and .md files
        all_docs = []
        
        # Load .txt files
        txt_loader = DirectoryLoader(
            str(source_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=False,
        )
        all_docs.extend(txt_loader.load())
        
        # Load .md files
        md_loader = DirectoryLoader(
            str(source_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=False,
        )
        all_docs.extend(md_loader.load())
        
        logger.info(f"Loaded {len(all_docs)} documents")
        return all_docs
    else:
        raise ValueError(f"Invalid source path: {source_path}")


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into chunks.
    
    Args:
        documents: List of documents
        chunk_size: Chunk size in characters (default: 500)
        chunk_overlap: Overlap between chunks (default: 50)
    
    Returns:
        List of document chunks
    """
    logger.info(f"Splitting {len(documents)} documents (size={chunk_size}, overlap={chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    splits = text_splitter.split_documents(documents)
    logger.info(f"Created {len(splits)} chunks")
    
    return splits


def create_rag_chain(
    retriever,
    llm,
    prompt_template: Optional[str] = None,
):
    """
    Create a RAG chain using LCEL (LangChain Expression Language).
    
    Args:
        retriever: LangChain retriever
        llm: LangChain LLM
        prompt_template: Custom prompt template
    
    Returns:
        Runnable chain
    """
    if prompt_template is None:
        prompt_template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs))
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


class SimpleRAG:
    """
    Simple RAG system using LangChain components.
    Config is passed explicitly, not read from files.
    
    Example:
        >>> rag = SimpleRAG()
        >>> rag.ingest("examples/demo-corpus/")
        >>> response = rag.query("What is machine learning?")
    """
    
    def __init__(
        self,
        embedding_type: str = "ollama",
        embedding_model: str = "embeddinggemma",
        llm_type: str = "ollama",
        llm_model: str = "llama3.2",
        llm_temperature: float = 0.7,
        vectorstore_type: str = "faiss",
    ):
        """
        Initialize RAG system.
        
        Args:
            embedding_type: Embedding type (ollama, openai, huggingface)
            embedding_model: Embedding model name
            llm_type: LLM type (ollama, openai, llamacpp)
            llm_model: LLM model name
            llm_temperature: LLM temperature
            vectorstore_type: Vector store type (faiss, chroma)
        """
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.vectorstore_type = vectorstore_type
        
        # Create components
        self.embeddings = get_embeddings(type=embedding_type, model=embedding_model)
        self.llm = get_llm(type=llm_type, model=llm_model, temperature=llm_temperature)
        
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        logger.info("SimpleRAG initialized")

    def _refresh_pipeline(self):
        """Ensure retriever and chain reflect the current vector store."""
        if self.vectorstore is None:
            self.retriever = None
            self.chain = None
            return
        current_k = 4
        if self.retriever and hasattr(self.retriever, "search_kwargs"):
            current_k = self.retriever.search_kwargs.get("k", current_k)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": current_k}
        )
        self.chain = create_rag_chain(self.retriever, self.llm)

    def _index_chunks(self, chunks: List[Document]):
        """Persist split document chunks without discarding existing data."""
        if not chunks:
            logger.warning("No document chunks to index; skipping")
            return
        if self.vectorstore is None:
            vectorstore_cls = get_vectorstore(self.embeddings, type=self.vectorstore_type)
            self.vectorstore = vectorstore_cls.from_documents(chunks, self.embeddings)
            logger.info(f"Vector store initialized with {len(chunks)} chunks")
        else:
            if not hasattr(self.vectorstore, "add_documents"):
                raise RuntimeError(
                    f"Vector store type '{self.vectorstore_type}' does not support incremental updates"
                )
            self.vectorstore.add_documents(chunks)
            logger.info(f"Appended {len(chunks)} chunks to existing vector store")
        self._refresh_pipeline()
    
    def ingest(
        self,
        source_path: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Ingest documents into vector store.
        
        Args:
            source_path: Path to file or directory
            chunk_size: Chunk size (default: 500)
            chunk_overlap: Chunk overlap (default: 50)
        """
        # Load documents
        documents = load_documents(source_path)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split documents
        splits = split_documents(documents, chunk_size, chunk_overlap)
        
        logger.info("Creating vector store and generating embeddings...")
        self._index_chunks(splits)
        logger.info(f"Ingestion complete: {len(splits)} chunks indexed")
    
    def ingest_documents(
        self,
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Ingest pre-created Document objects with metadata.
        
        Args:
            documents: List of LangChain Document objects with metadata
            chunk_size: Chunk size (default: 500)
            chunk_overlap: Chunk overlap (default: 50)
        """
        logger.info(f"Ingesting {len(documents)} documents with metadata")
        
        # Split documents (preserves metadata)
        splits = split_documents(documents, chunk_size, chunk_overlap)
        
        logger.info("Creating vector store and generating embeddings...")
        self._index_chunks(splits)
        logger.info(f"Ingestion complete: {len(splits)} chunks indexed")
    
    def query(
        self,
        question: str,
        k: int = 4,
    ) -> str:
        """
        Query the RAG system.
        
        Args:
            question: User question
            k: Number of documents to retrieve (default: 4)
        
        Returns:
            Generated answer
        """
        if self.chain is None:
            raise RuntimeError("Must call ingest() before query()")
        
        # Update retriever k if specified
        if k != 4:  # Only update if different from default
            self.retriever.search_kwargs["k"] = k
        
        logger.info(f"Querying: {question}")
        response = self.chain.invoke(question)
        
        return response
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """
        Retrieve relevant documents without generation.
        
        Args:
            query: User query
            k: Number of documents to retrieve (default: 4)
        
        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise RuntimeError("Must call ingest() before retrieve()")
        
        if k != 4:  # Only update if different from default
            self.retriever.search_kwargs["k"] = k
        
        docs = self.retriever.invoke(query)
        return docs
    
    def save(self, path: str | Path):
        """Save vector store to disk."""
        if self.vectorstore is None:
            raise RuntimeError("No vector store to save")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving vector store to {path}")
        self.vectorstore.save_local(str(path))
    
    def load(self, path: str | Path):
        """Load vector store from disk."""
        path = Path(path)
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        logger.info(f"Loading vector store from {path}")
        from langchain_community.vectorstores import FAISS
        self.vectorstore = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        self._refresh_pipeline()
        logger.info("Vector store loaded successfully")
