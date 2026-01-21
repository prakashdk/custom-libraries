# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-01-21

### Fixed
- **BREAKING FIX**: Metadata preservation in `ingest_from_texts()` and `add_document()` methods
  - Previously metadata parameter was accepted but not utilized
  - Now creates Document objects with metadata directly instead of writing temp files
- Added `ingest_documents()` method to SimpleRAG to handle Document objects with metadata

## [0.1.2] - 2026-01-21

### Added
- `free_query()` method: Direct LLM queries with custom context without retrieval
- `reindex_all()` method: Rebuild index with new chunk_size/chunk_overlap settings

## [0.1.1] - 2026-01-21

### Added
- Type hints support with py.typed marker file
- "Typing :: Typed" classifier for PEP 561 compliance

### Changed
- Consolidated README files (removed README-LIBRARY.md)

## [0.1.0] - 2026-01-10

### Added
- Initial project scaffold with library-ready architecture
- Core interfaces: `DocumentPipeline`, `EmbeddingService`, `IndexStore`, `ModelAdapter`, `Retriever`
- Configuration system with YAML defaults and environment variable overrides
- Structured codebase: `app/` (application-specific) and `common/` (library-candidate)
- CLI entrypoints: `ingest`, `reindex`, `serve`
- FastAPI server with `/health`, `/ingest`, `/query` endpoints
- Unit and integration test scaffolds with mocked adapters
- GitHub Actions CI pipeline for linting, type checking, and testing
- Comprehensive documentation: README, CONTRIBUTING, architecture guide
- Example demo corpus and usage instructions
- Bootstrap script for pyenv setup

### Documentation
- Detailed API contracts for all major interfaces
- Configuration reference with all keys and defaults
- Architecture decision notes on starter-kit vs. library separation
- Contributing guide with library extraction plan

### Developer Experience
- Black, isort, and mypy configuration
- pytest with markers for unit/integration tests
- pyenv-friendly setup with `.python-version`
- Bootstrap script for quick environment setup

[0.1.0]: https://github.com/ACME/llama-rag-starter/releases/tag/v0.1.0
