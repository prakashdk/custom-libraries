"""
tests/conftest.py

Shared pytest fixtures and configuration.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root_path():
    """Return project root path."""
    return project_root


@pytest.fixture
def config_path():
    """Return config directory path."""
    return project_root / "config"


@pytest.fixture
def examples_path():
    """Return examples directory path."""
    return project_root / "examples"
