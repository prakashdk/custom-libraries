"""
setup.py for llama-rag-lib

Pip-installable RAG library using LangChain and Ollama.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="llama_rag_lib",
    version="0.1.3",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple RAG library using LangChain and Ollama",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llama-rag-lib",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "llama_rag": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Typing :: Typed",
    ],
    python_requires=">=3.11",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-ollama>=0.1.0",
        "langchain-text-splitters>=0.0.1",
        "faiss-cpu>=1.7.4",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
)
