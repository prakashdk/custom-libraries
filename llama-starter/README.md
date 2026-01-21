# llama_rag_lib

A pip-installable RAG (Retrieval-Augmented Generation) library built with LangChain and Ollama.

[![PyPI](https://img.shields.io/pypi/v/llama_rag_lib)](https://pypi.org/project/llama_rag_lib/)
[![Python](https://img.shields.io/pypi/pyversions/llama_rag_lib)](https://pypi.org/project/llama_rag_lib/)

## Features

- **Simple API**: High-level `RAGService` for easy integration
- **LangChain-Based**: Uses battle-tested LangChain components
- **Local-First**: Works with Ollama (runs on your machine)
- **Flexible**: Support for multiple embedding and LLM providers
- **Production-Ready**: Sensible defaults, proper error handling

## Installation

```bash
# From PyPI (recommended)
pip install llama_rag_lib

# From TestPyPI (note: --extra-index-url for dependencies)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llama_rag_lib
```

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Models: `ollama pull embeddinggemma` and `ollama pull llama3.2`

## Quick Start

```python
from llama_rag import RAGService
from pathlib import Path

# Initialize service
service = RAGService(
    embedding_model="embeddinggemma",
    llm_model="llama3.2",
)

# Ingest documents
service.ingest_from_directory(Path("./docs"))

# Query
answer = service.query("What is RAG?")
print(answer)
```

## Usage Examples

### Basic Usage

```python
from llama_rag import RAGService

service = RAGService()
service.ingest_from_text("Python is a programming language.")
answer = service.query("What is Python?")
```

### Advanced Configuration

```python
from llama_rag import RAGService
from pathlib import Path

service = RAGService(
    # Model configuration
    embedding_type="ollama",
    embedding_model="embeddinggemma",
    llm_type="ollama",
    llm_model="llama3.2",
    llm_temperature=0.7,
    
    # Processing configuration
    chunk_size=500,
    chunk_overlap=50,
    retrieval_k=4,
    
    # Lifecycle
    index_path=Path("./my_index"),
    auto_save=True,
)

# Use as context manager
with service:
    service.ingest_from_directory(Path("./documents"))
    result = service.query("Your question")
```

### CLI Application

The package includes a CLI example in the `app/` folder:

```bash
# Clone repository for CLI
git clone https://github.com/yourusername/llama_rag_lib
cd llama_rag_lib

# Install library
pip install -e .

# Use CLI
python -m app.cli ingest examples/demo-corpus/ --output ./data/index
python -m app.cli query "What is ML?" --index ./data/index
```

## API Reference

### RAGService

```python
from llama_rag import RAGService

service = RAGService(
    index_path=Path("./data/index"),
    auto_load=True,
    auto_save=True,
    embedding_model="embeddinggemma",
    llm_model="llama3.2",
    chunk_size=500,
    retrieval_k=4,
)

# Ingestion
service.ingest_from_directory(Path("./docs"))
service.ingest_from_text("Custom text")
service.ingest_documents([doc1, doc2])

# Querying
answer = service.query("Question?")
docs = service.retrieve("Question?")
result = service.query_with_sources("Question?")

# Management
service.save()
service.load(Path("./index"))
service.has_index()
```

## Project Structure

```
llama_rag_lib/
├── src/llama_rag/      # Library package (pip-installable)
│   ├── service.py      # RAGService
│   ├── rag.py          # Core RAG implementation
│   ├── factories.py    # Component factories
│   ├── config.py       # Configuration
│   └── utils.py        # Utilities
├── app/                # CLI application (example)
├── tests/              # Test suite
└── examples/           # Usage examples
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Package

```bash
# Build distribution
python setup.py sdist

# Or use build module
pip install build
python -m build
```

### Publishing

```bash
# Install twine
pip install twine

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

See [PUBLISH-REMOTE.md](PUBLISH-REMOTE.md) for detailed publishing instructions.

## Configuration

The library uses sensible defaults but supports customization:

1. **Runtime parameters** (highest priority)
2. **Environment variables** (`LLAMA_RAG_*` prefix)
3. **YAML configuration** (`config/defaults.yaml`)
4. **Built-in defaults** (lowest priority)

See [docs/configuration.md](docs/configuration.md) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- PyPI: https://pypi.org/project/llama_rag_lib/
- TestPyPI: https://test.pypi.org/project/llama_rag_lib/
- Documentation: [docs/](docs/)
- Examples: [examples/](examples/)

## Support

For issues and questions, please visit the issue tracker.

```bash
# Run tests
pytest

# With coverage
pytest --cov=common --cov=app
```

## Project Structure

- **~230 lines**: `common/service.py` - Service layer
- **~220 lines**: `common/rag.py` - Core RAG implementation
- **~90 lines**: `common/factories.py` - Component factories
- **~160 lines**: `app/cli.py` - CLI interface
- **~700 lines total** (vs 2000+ with custom abstractions)

## Why This Approach?

**Before:** 2000+ lines of custom abstractions (EmbeddingService, ModelAdapter, IndexStore, etc.)

**After:** 700 lines using LangChain directly

**Benefits:**
- ✅ 93% less code to maintain
- ✅ Battle-tested LangChain components
- ✅ 100+ integrations available
- ✅ Easier to understand and extend
- ✅ Standard patterns developers know

## Documentation

- [CLI Usage Guide](examples/cli_usage.md) - Complete CLI examples
- [Configuration](docs/configuration.md) - Config reference
- [Architecture](docs/architecture.md) - Design decisions

## Dependencies

Core:
- `langchain` + ecosystem (core, community, ollama, text-splitters)
- `faiss-cpu` - Vector store
- `pyyaml` - Config

Dev:
- `pytest` + `pytest-cov` - Testing

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgments

- **LangChain** - For excellent abstractions
- **Ollama** - For local LLM support
- **FAISS** - For efficient vector search
