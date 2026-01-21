# 🚀 Quick Start Guide - llama-rag-lib

## Install

```bash
pip install -e .
```

> Need deterministic metadata search? Use `RecordsService` instead of `KnowledgeBaseService`. Both share the same ingestion APIs but maintain separate default indexes.

## Basic Usage

```python
from llama_rag import KnowledgeBaseService
from pathlib import Path

# Initialize
service = KnowledgeBaseService()

# Ingest
service.ingest_from_directory(Path("./docs"))

# Query
answer = service.query("What is this about?")
print(answer)
```

## Build Package

```bash
python -m build
```

## Install Built Package

```bash
pip install dist/llama_rag_lib-0.1.0-py3-none-any.whl
```

## Use CLI

```bash
python -m app.cli ingest examples/demo-corpus/ --output ./data/index
python -m app.cli query "What is ML?" --index ./data/index
```

## Run Tests

```bash
pytest tests/
```

## File Structure

```
src/llama_rag/     ← Your library (pip-installable)
app/               ← CLI application (uses library)
tests/             ← Tests
setup.py           ← Package setup
pyproject.toml     ← Project config
```

## Import Pattern

**Before:**
```python
from common.service import KnowledgeBaseService
```

**After:**
```python
from llama_rag import KnowledgeBaseService
```

## Documentation

- `INSTALL.md` - Installation details
- `README-LIBRARY.md` - Full API docs
- `CONVERSION-SUMMARY.md` - What changed
- `CHECKLIST.md` - Complete checklist
- `examples/library_usage.py` - Code examples

## Next Steps

1. Test: `pip install -e .`
2. Build: `python -m build`
3. Customize: Update author info in `setup.py`
4. Publish: `twine upload dist/*` (optional)

## Help

```bash
# Check if installed
pip show llama-rag-lib

# Test import
python -c "from llama_rag import KnowledgeBaseService; print('OK')"

# Run examples
python examples/library_usage.py
```

---

✅ **Library conversion complete!**
🎉 **Ready to use as pip package!**
