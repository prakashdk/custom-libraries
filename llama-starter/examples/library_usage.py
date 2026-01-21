"""
Example: Using llama-rag-lib in Your Python Projects

This script demonstrates how to use the llama-rag-lib library
after installing it with pip.
"""

from pathlib import Path
from llama_rag import RAGService


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage ===\n")
    
    # Initialize service with defaults
    service = RAGService(
        embedding_model="embeddinggemma",
        llm_model="llama3.2",
    )
    
    # Ingest documents
    docs_path = Path("./examples/demo-corpus")
    if docs_path.exists():
        count = service.ingest_from_directory(docs_path)
        print(f"✓ Ingested {count} documents")
    
    # Query
    answer = service.query("What is machine learning?")
    print(f"\nAnswer: {answer}\n")


def example_with_configuration():
    """Example with custom configuration."""
    print("=== Custom Configuration ===\n")
    
    service = RAGService(
        # Model configuration
        embedding_type="ollama",
        embedding_model="embeddinggemma",
        llm_type="ollama",
        llm_model="llama3.2",
        llm_temperature=0.7,
        vectorstore_type="faiss",
        
        # Processing configuration
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=4,
        
        # Lifecycle configuration
        index_path=Path("./data/custom_index"),
        auto_load=True,
        auto_save=True,
    )
    
    # Ingest text directly
    service.ingest_from_text(
        "RAG stands for Retrieval-Augmented Generation. "
        "It combines document retrieval with language generation."
    )
    
    # Query
    answer = service.query("What does RAG stand for?")
    print(f"Answer: {answer}\n")


def example_context_manager():
    """Example using context manager for auto-save."""
    print("=== Context Manager (Auto-Save) ===\n")
    
    index_path = Path("./data/temp_index")
    
    with RAGService(
        index_path=index_path,
        auto_save=True,
    ) as service:
        service.ingest_from_text(
            "Vector databases store embeddings for semantic search."
        )
        answer = service.query("What do vector databases store?")
        print(f"Answer: {answer}")
        # Index is automatically saved when exiting context
    
    print(f"✓ Index saved to {index_path}\n")


def example_retrieval_only():
    """Example: retrieve without generation."""
    print("=== Retrieval Without Generation ===\n")
    
    service = RAGService()
    
    # Ingest
    service.ingest_from_text(
        "Python is a high-level programming language. "
        "It is known for its simple syntax and readability."
    )
    
    # Retrieve documents without generating answer
    docs = service.retrieve("What is Python?")
    
    print("Retrieved documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:100]}...")
    print()


def example_query_with_sources():
    """Example: query with source documents."""
    print("=== Query With Sources ===\n")
    
    service = RAGService()
    
    # Ingest
    service.ingest_from_text(
        "LangChain is a framework for developing applications powered by language models."
    )
    
    # Query with sources
    result = service.query_with_sources("What is LangChain?")
    
    print(f"Answer: {result['answer']}\n")
    print("Sources:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"{i}. {doc.page_content[:100]}...")
    print()


def example_index_management():
    """Example: save and load index."""
    print("=== Index Management ===\n")
    
    index_path = Path("./data/managed_index")
    
    # Create and save
    service = RAGService(auto_save=False)
    service.ingest_from_text("Test document content")
    service.save(index_path)
    print(f"✓ Saved index to {index_path}")
    
    # Load existing
    service2 = RAGService(index_path=index_path, auto_load=True)
    print(f"✓ Loaded index from {index_path}")
    print(f"✓ Has index: {service2.has_index()}\n")


def example_using_factories():
    """Example: using lower-level factory functions."""
    print("=== Using Factory Functions ===\n")
    
    from llama_rag import get_embeddings, get_llm, get_vectorstore
    
    # Create components
    embeddings = get_embeddings(type="ollama", model="embeddinggemma")
    llm = get_llm(type="ollama", model="llama3.2", temperature=0.7)
    VectorStoreClass = get_vectorstore(embeddings, type="faiss")
    
    print(f"✓ Created embeddings: {type(embeddings).__name__}")
    print(f"✓ Created LLM: {type(llm).__name__}")
    print(f"✓ Got VectorStore class: {VectorStoreClass.__name__}\n")


def example_document_processing():
    """Example: document loading and splitting."""
    print("=== Document Processing ===\n")
    
    from llama_rag import load_documents, split_documents
    
    # Load documents
    docs_path = Path("./examples/demo-corpus")
    if docs_path.exists():
        docs = load_documents(docs_path)
        print(f"✓ Loaded {len(docs)} documents")
        
        # Split into chunks
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        print(f"✓ Split into {len(chunks)} chunks\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("llama-rag-lib Usage Examples")
    print("="*60 + "\n")
    
    # Run all examples
    try:
        example_basic_usage()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    try:
        example_with_configuration()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    try:
        example_context_manager()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    try:
        example_retrieval_only()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    try:
        example_query_with_sources()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    try:
        example_index_management()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    try:
        example_using_factories()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    try:
        example_document_processing()
    except Exception as e:
        print(f"⚠ Example skipped: {e}\n")
    
    print("="*60)
    print("Examples complete!")
    print("="*60 + "\n")
