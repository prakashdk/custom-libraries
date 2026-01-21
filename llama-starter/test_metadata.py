"""
Test metadata preservation in llama_rag_lib
"""

from pathlib import Path
from llama_rag import KnowledgeBaseService

# Initialize service
service = KnowledgeBaseService(
    embedding_model="embeddinggemma:latest",
    llm_model="llama3.2:latest",
    index_path=Path("data/test_index"),
    auto_save=False,  # Don't auto-save for testing
)

# Test documents with metadata
texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing deals with text and language."
]

metadata = [
    {"source": "ml-intro", "category": "fundamentals", "author": "Alice"},
    {"source": "deep-learning", "category": "advanced", "author": "Bob"},
    {"source": "nlp-guide", "category": "intermediate", "author": "Charlie"}
]

# Ingest with metadata
print("Ingesting documents with metadata...")
service.ingest_from_texts(texts, metadata=metadata)

# Query and check if metadata is preserved
print("\nQuerying for 'machine learning'...")
results = service.retrieve("machine learning", k=2)

print(f"\nRetrieved {len(results)} results:")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. Content: {doc.page_content[:100]}...")
    print(f"   Metadata: {doc.metadata}")
    
# Test add_document with metadata
print("\n\nTesting add_document with metadata...")
service.add_document(
    "Transformers are the foundation of modern NLP models.",
    metadata={"source": "transformer-guide", "category": "advanced", "author": "Dave"}
)

results = service.retrieve("transformers", k=1)
print(f"\nRetrieved result:")
print(f"Content: {results[0].page_content}")
print(f"Metadata: {results[0].metadata}")

print("\n✅ Metadata test complete!")
