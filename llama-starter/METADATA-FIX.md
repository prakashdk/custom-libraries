# Metadata Fix Summary - Version 0.2.0

## Issue
The `ingest_from_texts()` and `add_document()` methods accepted a `metadata` parameter but didn't use it properly. Metadata was being lost during ingestion.

## Root Cause
The `ingest_from_texts()` method was:
1. Writing text documents to temporary files
2. Calling `SimpleRAG.ingest()` with file paths
3. `ingest()` used `load_documents()` to load from files, which lost metadata

## Solution
1. **Added new method to SimpleRAG**: `ingest_documents(documents, chunk_size, chunk_overlap)`
   - Accepts pre-created Document objects with metadata
   - Preserves metadata through the entire pipeline

2. **Updated ingest_from_texts()**: 
   - Creates LangChain Document objects directly with metadata
   - Calls new `ingest_documents()` method instead of writing temp files
   - Each document gets its metadata attached properly

## Changes Made

### [src/llama_rag/rag.py](src/llama_rag/rag.py)
- Added `ingest_documents()` method that accepts Document objects

### [src/llama_rag/service.py](src/llama_rag/service.py)
- Updated `ingest_from_texts()` to create Document objects with metadata
- Metadata parameter is now properly utilized

## Testing
Created [test_metadata.py](test_metadata.py) to verify:
- ✅ Metadata preserved when using `ingest_from_texts()`
- ✅ Metadata preserved when using `add_document()`
- ✅ Metadata accessible via `retrieve()` method

Example:
```python
service.ingest_from_texts(
    texts=["Machine learning is AI subset."],
    metadata=[{"source": "ml-intro", "author": "Alice"}]
)

results = service.retrieve("machine learning")
print(results[0].metadata)  # {'source': 'ml-intro', 'author': 'Alice'}
```

## Version 0.2.0 Published
- Package: `llama_rag_lib==0.2.0`
- TestPyPI: https://test.pypi.org/project/llama-rag-lib/0.2.0/
- Install: `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llama_rag_lib==0.2.0`

## API Usage
```python
from llama_rag import KnowledgeBaseService
from pathlib import Path

service = KnowledgeBaseService(
    embedding_model="embeddinggemma:latest",
    llm_model="llama3.2:latest",
    index_path=Path("data/index")
)

# Add documents with metadata
texts = ["doc1 content", "doc2 content"]
metadata = [
    {"source": "file1.txt", "category": "tech"},
    {"source": "file2.txt", "category": "science"}
]

service.ingest_from_texts(texts, metadata=metadata)

# Retrieve with metadata
results = service.retrieve("query", k=2)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```
