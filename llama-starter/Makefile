.PHONY: help install dev-install test lint format build clean publish-test publish bump-patch bump-minor bump-major

help:
	@echo "llama-rag-lib - Build & Publish Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install package locally"
	@echo "  make dev-install   - Install with dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo ""
	@echo "Building:"
	@echo "  make build         - Build distribution packages"
	@echo "  make clean         - Remove build artifacts"
	@echo ""
	@echo "Versioning:"
	@echo "  make bump-patch    - Bump patch version (0.1.0 → 0.1.1)"
	@echo "  make bump-minor    - Bump minor version (0.1.0 → 0.2.0)"
	@echo "  make bump-major    - Bump major version (0.1.0 → 1.0.0)"
	@echo ""
	@echo "Publishing:"
	@echo "  make publish-test  - Publish to TestPyPI"
	@echo "  make publish       - Publish to PyPI"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pip install build twine

test:
	pytest tests/ -v

lint:
	ruff check src/llama_rag tests/ || true

format:
	black src/llama_rag tests/ || true
	isort src/llama_rag tests/ || true

build: clean
	python -m build
	@echo ""
	@echo "✅ Built packages in dist/"
	@echo ""
	@echo "To install in another project:"
	@echo "  pip install $$(pwd)/dist/*.whl"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

bump-patch:
	python scripts/bump_version.py patch

bump-minor:
	python scripts/bump_version.py minor

bump-major:
	python scripts/bump_version.py major

publish-test: build
	twine upload --repository testpypi dist/*
	@echo ""
	@echo "✓ Published to TestPyPI"
	@echo "Test installation:"
	@echo "  pip install --index-url https://test.pypi.org/simple/ llama-rag-lib"

publish: build
	twine upload dist/*
	@echo ""
	@echo "✓ Published to PyPI"
	@echo "Install with:"
	@echo "  pip install llama-rag-lib"

quick-patch: bump-patch build
	@echo ""
	@echo "Ready to publish! Run:"
	@echo "  make publish-test  # Test on TestPyPI first"
	@echo "  make publish       # Publish to PyPI"
