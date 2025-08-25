# Arges - LLM-Powered Application

An intelligent application powered by Large Language Models (LLM) for advanced
natural language processing tasks.

## Features

- LLM integration with OpenAI API
- Command-line interface
- Configuration management
- Comprehensive testing suite
- Type safety with Pydantic
- Beautiful terminal output with Rich

## Installation

### Development Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd arges
```

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install dependencies:

```bash
pip install -e ".[dev]"
```

1. Set up pre-commit hooks:

```bash
pre-commit install
```

1. Copy environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Command Line Interface

```bash
# Basic usage
arges --help

# Example command
arges process "Your text here"
```

### Python API

```python
from arges import LLMClient

client = LLMClient()
response = client.process("Your text here")
print(response)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

### Documentation

```bash
# Serve documentation locally
mkdocs serve
```

## Project Structure

```text
arges/
├── src/
│   └── arges/
│       ├── __init__.py
│       ├── cli.py
│       ├── client.py
│       ├── config.py
│       └── models.py
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   └── test_cli.py
├── docs/
├── config/
├── scripts/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
