#!/usr/bin/env bash

# Bootstrap script for LLaMA RAG Starter Kit
# This script sets up the development environment using pyenv

set -e  # Exit on error

echo "========================================="
echo "LLaMA RAG Starter Kit - Bootstrap Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo -e "${YELLOW}pyenv is not installed.${NC}"
    echo "Please install pyenv first:"
    echo ""
    echo "  macOS (Homebrew):"
    echo "    brew install pyenv"
    echo ""
    echo "  Linux:"
    echo "    curl https://pyenv.run | bash"
    echo ""
    echo "Then add to your shell profile:"
    echo '  export PYENV_ROOT="$HOME/.pyenv"'
    echo '  export PATH="$PYENV_ROOT/bin:$PATH"'
    echo '  eval "$(pyenv init -)"'
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ pyenv found${NC}"

# Check Python version from .python-version
PYTHON_VERSION=$(cat .python-version)
echo "Required Python version: ${PYTHON_VERSION}"

# Check if Python version is installed
if ! pyenv versions | grep -q "${PYTHON_VERSION}"; then
    echo -e "${YELLOW}Python ${PYTHON_VERSION} not installed. Installing...${NC}"
    pyenv install "${PYTHON_VERSION}"
else
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} already installed${NC}"
fi

# Set local Python version
pyenv local "${PYTHON_VERSION}"

# Check if virtual environment exists
VENV_NAME="llama-rag-env"

if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
else
    echo "Creating virtual environment..."
    python -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Bootstrap complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Try the demo:"
echo "   python -m app.cli ingest examples/demo-corpus/ --output ./data/index"
echo ""
echo "3. Start the server:"
echo "   python -m app.main"
echo ""
echo "4. Run tests:"
echo "   pytest tests/"
echo ""
echo "For more information, see README.md"
