# CLI Usage Examples

This document shows how to use the RAG CLI for document ingestion and querying.

## Quick Start

### Enable Verbose Logging

By default, the CLI provides clean, minimal output. Use `--verbose` or `-v` for detailed process logs:

```bash
# Clean output (default)
python -m app.cli ingest examples/demo-corpus/ --output ./data/index

# Verbose output with timestamps
python -m app.cli --verbose ingest examples/demo-corpus/ --output ./data/index
```

## Basic Usage

### 1. Ingest Documents

```bash
# Ingest from demo corpus
python -m app.cli ingest examples/demo-corpus/ --output ./data/index

# Ingest with custom chunking
python -m app.cli ingest examples/demo-corpus/ \
  --output ./data/index \
  --chunk-size 1000 \
  --overlap 100

# Ingest multiple directories
python -m app.cli ingest corpus1/ corpus2/ docs/ --output ./data/combined-index

# Verbose mode to see detailed progress
python -m app.cli --verbose ingest docs/ --output ./data/index
```

### 2. Query Documents

```bash
# Simple query
python -m app.cli query "What is machine learning?" --index ./data/index

# Query with more retrieved documents
python -m app.cli query "Explain deep learning" --index ./data/index --k 10

# Show retrieved documents
python -m app.cli query "What is NLP?" --index ./data/index --show-docs

# Verbose mode to see retrieval process
python -m app.cli --verbose query "What is AI?" --index ./data/index
```

## Complete Workflow Example

### Step 1: Prepare Documents

Create a directory with your text files:

```bash
mkdir my-documents
echo "Machine learning is a subset of AI." > my-documents/ml.txt
echo "Deep learning uses neural networks." > my-documents/dl.txt
echo "NLP helps computers understand language." > my-documents/nlp.txt
```

### Step 2: Ingest

**Default (clean output):**
```bash
python -m app.cli ingest my-documents/ --output ./data/my-index
```

Output:
```
📥 Ingesting documents from: my-documents/
✅ Ingestion complete! Index saved to: ./data/my-index
```

**Verbose mode (detailed logs):**
```bash
python -m app.cli --verbose ingest my-documents/ --output ./data/my-index
```

Output:
```
12:34:56 - llama_rag.service - INFO - Ingesting from directory: my-documents/
12:34:57 - llama_rag.rag - INFO - Loaded 3 documents
12:34:57 - llama_rag.rag - INFO - Splitting 3 documents (size=500, overlap=50)
12:34:57 - llama_rag.rag - INFO - Created 3 chunks
12:34:58 - llama_rag.rag - INFO - Creating vector store and generating embeddings...
12:35:02 - llama_rag.rag - INFO - Ingestion complete: 3 chunks indexed
✅ Ingestion complete! Index saved to: ./data/my-index
```

### Step 3: Query

**Default (clean output):**
```bash
python -m app.cli query "What is machine learning?" --index ./data/my-index
```

Output:
```
🔍 Querying: What is machine learning?

================================================================================
ANSWER:
================================================================================
Machine learning is a subset of artificial intelligence (AI) that focuses on...
================================================================================
```

**Verbose mode (show retrieval process):**
```bash
python -m app.cli --verbose query "What is machine learning?" --index ./data/my-index
```

### Step 4: Query with Source Documents

```bash
python -m app.cli query "Tell me about NLP" --index ./data/my-index --show-docs
```

Output:
```
================================================================================
ANSWER:
================================================================================
NLP (Natural Language Processing) helps computers understand human language...
================================================================================

RETRIEVED DOCUMENTS:
================================================================================

[1] my-documents/nlp.txt
NLP helps computers understand language...

[2] my-documents/ml.txt
Machine learning is a subset of AI...
================================================================================
```

## Advanced Usage

### Custom Chunking Strategy

For longer documents, adjust chunk size and overlap:

```bash
# Large chunks for context-heavy tasks
python -m app.cli ingest docs/ \
  --output ./data/index \
  --chunk-size 1500 \
  --overlap 200

# Small chunks for precise retrieval
python -m app.cli ingest docs/ \
  --output ./data/index \
  --chunk-size 300 \
  --overlap 30
```

### Multiple Sources

Ingest from multiple directories:

```bash
python -m app.cli ingest \
  technical-docs/ \
  user-guides/ \
  faqs/ \
  --output ./data/knowledge-base
```

### Different Indexes

Maintain separate indexes for different domains:

```bash
# Technical documentation
python -m app.cli ingest tech-docs/ --output ./data/tech-index

# Marketing content
python -m app.cli ingest marketing/ --output ./data/marketing-index

# Query specific index
python -m app.cli query "API documentation" --index ./data/tech-index
```

## Programmatic Usage

You can also use the service directly in Python:

### Basic Usage

```python
from common.service import RAGService
from common.utils import configure_logging
from pathlib import Path

# Optional: enable verbose logging
configure_logging(verbose=True)

# Create service
service = RAGService(index_path=Path("./data/index"))

# Ingest
service.ingest_from_directory(Path("my-documents/"))

# Query
answer = service.query("What is machine learning?")
print(answer)

# Retrieve documents
docs = service.retrieve("deep learning", k=5)
for doc in docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content}\n")
```

### Using Context Manager (Auto-save)

The recommended pattern for ensuring your index is saved:

```python
from common.service import RAGService
from pathlib import Path

# Auto-saves on exit
with RAGService(
    index_path=Path("./data/index"),
    auto_load=True,
    auto_save=True
) as service:
    # Ingest documents
    service.ingest_from_directory(Path("docs/"))
    
    # Query
    answer = service.query("What is AI?")
    print(answer)

# Index automatically saved here
```

### Advanced Configuration

```python
from common.service import RAGService
from pathlib import Path

service = RAGService(
    # Index settings
    index_path=Path("./data/custom-index"),
    auto_load=False,
    auto_save=True,
    
    # Model configuration
    embedding_type="ollama",
    embedding_model="embeddinggemma",
    llm_type="ollama",
    llm_model="llama3.1",
    llm_temperature=0.5,
    
    # Processing configuration
    chunk_size=1000,
    chunk_overlap=100,
    retrieval_k=8,
)

# Custom chunking for specific corpus
service.ingest_from_directory(
    Path("technical-docs/"),
    chunk_size=1500,  # Override for this corpus
    chunk_overlap=150,
)

# Query with custom k
answer = service.query("Explain the architecture", k=10)
```

### Error Handling

The service provides helpful error messages:

```python
from common.service import RAGService
from pathlib import Path

try:
    service = RAGService(index_path=Path("./data/index"))
    answer = service.query("What is ML?")
except ValueError as e:
    # No index loaded
    print(f"Error: {e}")
    # Shows: "No index available. Please ingest documents first..."
except RuntimeError as e:
    # Ollama not running
    print(f"Error: {e}")
    # Shows: "Cannot connect to Ollama. Please ensure Ollama is running..."
```

## Tips

### 1. Chunk Size Guidelines

- **Technical documentation**: 800-1500 tokens (more context)
- **General content**: 400-800 tokens (balanced)
- **Short Q&A**: 200-400 tokens (precise)

### 2. Retrieval K

- **Comprehensive answers**: k=8-10
- **Balanced**: k=4-6 (default)
- **Quick/focused**: k=2-3

### 3. Index Management

```bash
# Check index exists
ls -la ./data/index/

# Index contents
./data/index/
  ├── index.faiss    # Vector index
  └── index.pkl      # Document metadata

# Backup index
cp -r ./data/index ./data/index.backup

# Remove index to start fresh
rm -rf ./data/index
```

## Troubleshooting

### Index Not Found

```
ERROR: Index not found: ./data/index
ERROR: Run 'ingest' command first to create an index
```

**Solution**: Run ingest first:
```bash
python -m app.cli ingest examples/demo-corpus/ --output ./data/index
```

### No Documents Loaded

```
INFO: Loaded 0 documents
```

**Solution**: Check directory path and file extensions (supports .txt, .md)

### Empty Response

If you get generic/empty answers:
- Try increasing k: `--k 8`
- Check if documents are relevant
- Re-ingest with better chunking

## Environment Variables

Override config via environment:

```bash
# Custom index path
export LLAMA_RAG_INDEX__PATH=./my-custom-index

# Run with custom config
python -m app.cli query "test" --index $LLAMA_RAG_INDEX__PATH
```

## Next Steps

- Try with your own documents
- Experiment with different chunk sizes
- Build scripts that use RAGService programmatically
- Integrate into your applications
