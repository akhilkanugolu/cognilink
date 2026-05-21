# Contributing to CogniLink

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/cognilink/cognilink.git
cd cognilink
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

3. Install in development mode:
```bash
pip install -e ".[dev,pdf]"
```

4. Run tests:
```bash
pytest tests/ -x -q
```

5. Run linting:
```bash
ruff check cognilink/
ruff format cognilink/
```

## Project Structure

```
cognilink/
├── __init__.py          # Package entry point with map() function
├── orchestrator.py      # SessionOrchestrator — top-level coordinator
├── core/
│   ├── models.py        # Data models (Node, Edge, SessionConfig, etc.)
│   ├── graph_store.py   # SQLite graph storage engine
│   ├── cascader.py      # Mutation cascade propagation
│   └── exceptions.py    # Custom exceptions
├── extract/
│   ├── registry.py      # ExtractorRegistry — plugin-based file parsers
│   └── pdf.py           # PDF parser using PyPDF2
├── inference/
│   └── provider.py      # invoke_llm() — thin LangChain wrapper
└── viz/
    ├── html_renderer.py # Visualizer — HTML + JSON output
    ├── json_export.py   # Standalone JSON export
    └── templates/
        └── graph.html   # Jinja2 template with vis.js
```

## Running Tests

```bash
# All tests
pytest tests/ -x -q

# Specific test file
pytest tests/unit/test_graph_store.py -v

# With coverage
pytest tests/ --cov=cognilink --cov-report=term-missing
```

## Code Style

- We use `ruff` for linting and formatting
- Type hints are required for all public functions
- Docstrings follow Google style
- Keep it simple — avoid unnecessary abstractions

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass: `pytest tests/ -x -q`
4. Ensure linting passes: `ruff check cognilink/`
5. Submit a PR with a clear description of changes

## Architecture Principles

- **LangChain-native**: Users bring their own LangChain model. No custom LLM abstractions.
- **Zero-token extraction**: Document parsing is purely programmatic.
- **Atomic operations**: All graph mutations are wrapped in SQLite transactions.
- **Security by default**: No raw content in outputs, parameterized SQL only.
