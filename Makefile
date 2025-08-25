# Makefile for Arges LLM Application
.PHONY: help install install-dev test test-cov lint format clean build docs serve-docs setup venv activate venv-info lint-python lint-markdown format-python format-markdown type-check pre-commit run run-chat run-example config env-check install-hooks profile security-check update-deps freeze dev ci init-data lines size reset version bump-patch bump-minor bump-major

# Virtual environment detection
VENV_PATH := .venv
VENV_BIN := $(VENV_PATH)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
ACTIVATE := $(VENV_BIN)/activate

# Check if virtual environment exists
VENV_EXISTS := $(shell test -d $(VENV_PATH) && echo 1 || echo 0)

# Use virtual environment if it exists, otherwise use system Python
ifeq ($(VENV_EXISTS),1)
    PYTHON_CMD := $(PYTHON)
    PIP_CMD := $(PIP)
    RUN_IN_VENV := . $(ACTIVATE) &&
else
    PYTHON_CMD := python3
    PIP_CMD := pip3
    RUN_IN_VENV :=
endif

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Virtual environment
venv: ## Create virtual environment
	$(PYTHON_CMD) -m venv $(VENV_PATH)
	@echo "Virtual environment created. Activate with: source $(ACTIVATE)"

activate: ## Show command to activate virtual environment
	@if [ $(VENV_EXISTS) -eq 1 ]; then \
		echo "Virtual environment exists. To activate, run:"; \
		echo "source $(ACTIVATE)"; \
	else \
		echo "Virtual environment not found. Create one with: make venv"; \
	fi

venv-info: ## Show virtual environment information
	@echo "Virtual environment path: $(VENV_PATH)"
	@echo "Virtual environment exists: $(VENV_EXISTS)"
	@if [ $(VENV_EXISTS) -eq 1 ]; then \
		echo "Python executable: $(PYTHON)"; \
		echo "Pip executable: $(PIP)"; \
		echo "To activate: source $(ACTIVATE)"; \
	else \
		echo "Virtual environment not found. Create with: make venv"; \
	fi

# Installation
install: ## Install the package in production mode
	$(RUN_IN_VENV) $(PIP_CMD) install .

install-dev: ## Install the package in development mode with all dependencies
	$(RUN_IN_VENV) $(PIP_CMD) install --upgrade pip
	$(RUN_IN_VENV) $(PIP_CMD) install -e ".[dev,docs]"
	npm install
	$(RUN_IN_VENV) pre-commit install

setup: venv install-dev ## Complete development environment setup
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file - please edit with your API keys"; fi
	@echo "Setup complete! Don't forget to activate the virtual environment: source $(ACTIVATE)"

# Testing
test: ## Run tests
	$(RUN_IN_VENV) pytest

test-cov: ## Run tests with coverage report
	$(RUN_IN_VENV) pytest --cov=src --cov-report=term-missing --cov-report=html

test-watch: ## Run tests in watch mode
	$(RUN_IN_VENV) pytest-watch

# Code quality
lint: ## Run all linters (Python and Markdown)
	$(RUN_IN_VENV) ruff check src tests
	$(RUN_IN_VENV) mypy src
	$(RUN_IN_VENV) black --check src tests
	npx markdownlint-cli2 "**/*.md"

lint-python: ## Run Python linters only
	$(RUN_IN_VENV) ruff check src tests
	$(RUN_IN_VENV) mypy src
	$(RUN_IN_VENV) black --check src tests

lint-markdown: ## Run Markdown linters only
	npx markdownlint-cli2 "**/*.md"

format: ## Format all code (Python and Markdown)
	$(RUN_IN_VENV) ruff format src tests
	$(RUN_IN_VENV) black src tests
	npx prettier --write "**/*.md"

format-python: ## Format Python code only
	$(RUN_IN_VENV) ruff format src tests
	$(RUN_IN_VENV) black src tests
	$(RUN_IN_VENV) ruff check --fix src tests

format-markdown: ## Format Markdown files only
	npx markdownlint-cli2 --fix "**/*.md"
	npx prettier --write "**/*.md"

type-check: ## Run type checking with mypy
	$(RUN_IN_VENV) mypy src

# Pre-commit
pre-commit: ## Run pre-commit hooks on all files
	$(RUN_IN_VENV) pre-commit run --all-files

# Documentation
docs: ## Build documentation
	$(RUN_IN_VENV) mkdocs build

serve-docs: ## Serve documentation locally
	$(RUN_IN_VENV) mkdocs serve

# Building and publishing
build: clean ## Build distribution packages
	$(RUN_IN_VENV) $(PYTHON_CMD) -m build

clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development commands
run: ## Run the CLI application (requires arguments)
	$(RUN_IN_VENV) $(PYTHON_CMD) -m arges

run-chat: ## Start interactive chat session
	$(RUN_IN_VENV) $(PYTHON_CMD) -m arges chat

run-example: ## Run example command
	$(RUN_IN_VENV) $(PYTHON_CMD) -m arges process "Hello, how are you?" --model gpt-3.5-turbo

# Configuration and environment
config: ## Show current configuration
	$(RUN_IN_VENV) $(PYTHON_CMD) -m arges config-info

env-check: ## Check if environment variables are set
	@echo "Checking environment variables..."
	@$(RUN_IN_VENV) $(PYTHON_CMD) -c "from arges.config import config; print('✓ Configuration loaded successfully')" || echo "✗ Configuration error - check your .env file"

# Docker commands (optional - if you add Docker support later)
docker-build: ## Build Docker image
	docker build -t arges:latest .

docker-run: ## Run application in Docker
	docker run --rm -it arges:latest

# Git hooks
install-hooks: ## Install git hooks
	$(RUN_IN_VENV) pre-commit install
	$(RUN_IN_VENV) pre-commit install --hook-type commit-msg

# Performance and profiling
profile: ## Run with profiler
	$(RUN_IN_VENV) $(PYTHON_CMD) -m cProfile -o profile.stats -m arges process "Test profiling"

# Security
security-check: ## Run security checks
	$(RUN_IN_VENV) bandit -r src/
	$(RUN_IN_VENV) safety check

# Dependencies
update-deps: ## Update all dependencies
	$(RUN_IN_VENV) pip-compile --upgrade pyproject.toml
	$(RUN_IN_VENV) pip-sync

freeze: ## Freeze current dependencies
	$(RUN_IN_VENV) pip freeze > requirements-frozen.txt

# Quick development workflow
dev: clean install-dev test lint ## Full development workflow: clean, install, test, lint

# CI/CD simulation
ci: clean install-dev test-cov lint ## Simulate CI pipeline

# Database/Data commands (extend as needed)
init-data: ## Initialize data directories
	mkdir -p data/raw data/processed data/models

# Utilities
lines: ## Count lines of code
	find src -name "*.py" | xargs wc -l

size: ## Show project size
	du -sh .

# Emergency commands
reset: clean ## Reset project to clean state
	rm -rf $(VENV_PATH)/
	rm -f .env
	@echo "Project reset. Run 'make setup' to reinitialize."

# Version management
version: ## Show current version
	$(RUN_IN_VENV) $(PYTHON_CMD) -c "from arges import __version__; print(__version__)"

bump-patch: ## Bump patch version
	$(RUN_IN_VENV) bumpversion patch

bump-minor: ## Bump minor version
	$(RUN_IN_VENV) bumpversion minor

bump-major: ## Bump major version
	$(RUN_IN_VENV) bumpversion major
