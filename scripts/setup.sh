#!/bin/bash

# Setup script for Arges development environment

set -e

echo "Setting up Arges development environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check Python version (require 3.9+)
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "Python version $python_version is supported"
else
    echo "Error: Python 3.9+ is required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing Arges in development mode..."
pip install -e ".[dev,docs]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys"
fi

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  arges --help"
echo ""
echo "Don't forget to edit .env with your OpenAI API key!"
