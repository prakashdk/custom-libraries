# Contributing to LLaMA RAG Starter Kit

Thank you for your interest in contributing! This guide covers development workflow, code standards, and the plan for extracting `common/` into an internal library.

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Library Extraction Plan](#library-extraction-plan)

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/your-org/llama-starter.git
cd llama-starter
```

### 2. Set Up Environment

```bash
# Use pyenv for Python version management
pyenv install 3.11.6
pyenv local 3.11.6

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies including dev tools
pip install -r requirements.txt
```

### 3. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linting and formatting before commits.

## 📝 Code Standards

### Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Import sorting**: Use `isort` with Black-compatible profile
- **Formatting**: Use `black` for consistent code formatting
- **Type hints**: Encouraged but not required for all functions

### Running Formatters

```bash
# Format code
black .

# Sort imports
isort .

# Check formatting without changes
black --check .
isort --check .
```

### Type Checking

```bash
# Run mypy on common and app modules
mypy common/ app/ --ignore-missing-imports
```

### Linting

```bash
# Run flake8 (optional)
flake8 common/ app/ --max-line-length=100
```

### Documentation Standards

**Module Docstrings**: Every module should have a docstring explaining its purpose.

```python
"""
common/example.py

Brief description of what this module does.

Longer description if needed, including design notes,
usage examples, or important considerations.
"""
```

**Function/Class Docstrings**: Use Google-style docstrings.

```python
def process_documents(sources: List[str], chunk_size: int = 512) -> List[Doc]:
    """
    Process documents from sources into Doc objects.
    
    Args:
        sources: List of file or directory paths
        chunk_size: Target tokens per chunk
    
    Returns:
        List of processed Doc objects
    
    Raises:
        FileNotFoundError: If source doesn't exist
        ValueError: If chunk_size is invalid
    
    Example:
        >>> docs = process_documents(["data/"], chunk_size=1024)
        >>> len(docs)
        42
    """
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=common --cov=app --cov-report=html

# Run specific test file
pytest tests/test_pipeline_unit.py -v

# Run tests with specific marker
pytest tests/ -m unit
pytest tests/ -m integration
```

### Test Organization

- `tests/test_*_unit.py`: Unit tests with mocks
- `tests/test_interfaces.py`: Interface contract tests
- `tests/integration_*.py`: Integration tests

### Writing Tests

**Unit Tests**: Test individual functions/classes in isolation.

```python
def test_doc_checksum_generation():
    """Test that Doc objects generate consistent checksums."""
    doc = Doc(text="test content")
    assert doc.checksum.startswith("sha256:")
    
    # Same content should produce same checksum
    doc2 = Doc(text="test content")
    assert doc.checksum == doc2.checksum
```

**Integration Tests**: Test component interactions.

```python
def test_end_to_end_retrieval(mock_embedder, sample_docs):
    """Test complete retrieval workflow."""
    # Set up components
    index = InMemoryIndexStore(dimension=128)
    retriever = Retriever(mock_embedder, index)
    
    # Add documents
    # ... test logic ...
    
    # Verify results
    assert len(results) > 0
```

### Test Fixtures

Use pytest fixtures for reusable test data:

```python
@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Doc(text="Document 1", metadata={"id": 1}),
        Doc(text="Document 2", metadata={"id": 2}),
    ]
```

## 📤 Submitting Changes

### Branch Naming

- `feature/short-description` - New features
- `fix/bug-description` - Bug fixes
- `docs/what-changed` - Documentation only
- `refactor/what-changed` - Code refactoring
- `test/what-added` - Test additions

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer explanation if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build/tooling changes

Examples:
```
feat(embeddings): add Cohere embedding adapter

Implements EmbeddingService interface for Cohere API.
Includes retry logic and rate limiting.

Closes #45
```

```
fix(pipeline): handle empty files gracefully

Previously crashed on empty files. Now skips with warning.

Fixes #67
```

### Pull Request Process

1. **Create a branch** from `develop`
2. **Make changes** following code standards
3. **Add tests** for new functionality
4. **Run full test suite** and ensure all pass
5. **Update documentation** if needed
6. **Create pull request** with description of changes
7. **Address review comments**
8. **Squash merge** once approved

### Pull Request Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
```

## 📦 Library Extraction Plan

The `common/` module is designed to be extracted into a standalone internal pip library. Here's the detailed plan:

### Phase 1: Preparation (Current)

**Status**: ✅ Complete

- [x] Separate `common/` (library code) from `app/` (application code)
- [x] Define stable interfaces (EmbeddingService, IndexStore, etc.)
- [x] Comprehensive docstrings and type hints
- [x] Unit and integration tests
- [x] No dependencies on `app/` or config files

### Phase 2: Package Structure

**Timeline**: When first extraction is needed

1. Create new repository `llama-rag-lib/`

2. Copy `common/` directory structure:
   ```
   llama-rag-lib/
   ├── pyproject.toml
   ├── README.md
   ├── LICENSE
   ├── llama_rag/           # Renamed from common/
   │   ├── __init__.py
   │   ├── pipeline.py
   │   ├── embeddings.py
   │   ├── index.py
   │   ├── model_adapter.py
   │   ├── retriever.py
   │   ├── models.py
   │   ├── config.py
   │   ├── tokenizer_utils.py
   │   └── utils.py
   └── tests/
   ```

3. Update `pyproject.toml` for library:
   ```toml
   [project]
   name = "llama-rag-lib"
   version = "0.1.0"
   description = "Reusable RAG components for internal projects"
   
   dependencies = [
       "numpy>=1.24.0",
       "faiss-cpu>=1.7.4",
       "sentence-transformers>=2.2.0",
       "pydantic>=2.0.0",
       "pyyaml>=6.0",
   ]
   ```

### Phase 3: Library Publishing

**Setup Internal PyPI** (one-time):

```bash
# Option 1: AWS CodeArtifact
aws codeartifact create-repository \
    --domain company-domain \
    --repository llama-rag-lib

# Option 2: Self-hosted PyPI (devpi, artifactory)
# Follow your organization's process
```

**Build and Publish**:

```bash
cd llama-rag-lib/

# Build wheel
python -m build

# Publish to internal PyPI
twine upload \
    --repository-url https://pypi.internal.company.com \
    dist/*
```

### Phase 4: Update Starter Kit

In `llama-starter/`:

1. Update `pyproject.toml`:
   ```toml
   dependencies = [
       "llama-rag-lib>=0.1.0",  # Add library
       # Remove dependencies now in library
   ]
   ```

2. Update imports:
   ```python
   # Before
   from common.pipeline import DocumentPipeline
   
   # After
   from llama_rag.pipeline import DocumentPipeline
   ```

3. Remove `common/` directory

4. Update documentation

### Phase 5: Versioning Strategy

Use semantic versioning (SemVer):

- **0.x.y**: Pre-1.0, breaking changes allowed
- **1.0.0**: First stable release with API guarantees
- **1.x.y**: Backward-compatible changes
- **2.0.0**: Breaking changes

**Version Bumps**:
- `0.1.0` → `0.1.1`: Bug fixes
- `0.1.0` → `0.2.0`: New features (backward compatible)
- `0.1.0` → `1.0.0`: First stable API
- `1.0.0` → `2.0.0`: Breaking changes

### Phase 6: Multi-Project Usage

Once extracted, use in multiple projects:

**Project A** (RAG Chatbot):
```toml
dependencies = ["llama-rag-lib>=0.1.0"]
```

**Project B** (Document Search):
```toml
dependencies = ["llama-rag-lib>=0.1.0"]
```

**Benefits**:
- Shared bug fixes and improvements
- Consistent RAG implementation across projects
- Faster development for new RAG projects
- Centralized maintenance

### Interface Stability Guarantees

**Stable Interfaces** (won't break in minor versions):
- `Doc` model fields
- `EmbeddingService.embed_texts()` signature
- `IndexStore.add()`, `.search()` signatures
- `ModelAdapter.generate()` signature
- `Retriever.retrieve()` signature

**May Change** (until 1.0.0):
- Internal implementation details
- Config structure
- Utility functions
- Performance characteristics

### Migration Checklist

When extracting to library:

- [ ] Copy `common/` to new repo
- [ ] Set up packaging (`pyproject.toml`, `setup.py`)
- [ ] Configure internal PyPI
- [ ] Build and publish first version
- [ ] Update starter kit dependencies
- [ ] Update all import statements
- [ ] Remove `common/` from starter kit
- [ ] Update documentation
- [ ] Test with clean virtual environment
- [ ] Tag release in both repos

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the best solution for the project
- Help newcomers get started

### Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Open an issue with reproduction steps
- **Features**: Open an issue with use case description
- **Security**: Email security@company.com (private disclosure)

### Areas for Contribution

High-impact areas:

1. **Additional Adapters**
   - Embedding: Cohere, Voyage, Together AI
   - Index: Pinecone, Weaviate, Qdrant
   - Model: Claude, Gemini, open-source alternatives

2. **Advanced Features**
   - Hybrid search (vector + keyword)
   - Reranking with cross-encoders
   - Query understanding and expansion
   - Conversation history management

3. **Performance**
   - Batch processing optimizations
   - Caching strategies
   - Async/await support
   - GPU acceleration

4. **Documentation**
   - More usage examples
   - Architecture deep-dives
   - Performance tuning guides
   - Common pitfalls and solutions

Thank you for contributing! 🎉
