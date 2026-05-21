# CogniLink

**Document-to-graph knowledge mapper powered by LangChain.**

CogniLink ingests documents (PDFs, text files), builds a directed knowledge graph with LLM-generated summaries, and outputs interactive HTML visualizations. When upstream documents change, downstream nodes are automatically flagged as stale — keeping your knowledge graph consistent.

## Features

- **Simple API** — one function call to go from documents to knowledge graph
- **Any LLM** — bring your own LangChain model (OpenAI, Anthropic, Ollama, Bedrock, etc.)
- **Mutation cascading** — update a document and all dependents are flagged stale
- **Interactive visualization** — clickable HTML graph with vis.js
- **Dual storage** — ephemeral in-memory or persistent SQLite
- **Zero-token extraction** — document parsing is purely programmatic

## Installation

```bash
pip install cognilink[pdf]
```

For development:
```bash
pip install cognilink[dev]
```

## Quick Start

```python
import cognilink
from langchain_openai import ChatOpenAI

# One-liner: document → knowledge graph
result = cognilink.map(
    "./docs/spec.pdf",
    model=ChatOpenAI(api_key="sk-...")
)

print(result["nodes"])      # List of node dicts
print(result["html_path"])  # Path to interactive HTML visualization
```

### Multiple documents with relationships

```python
result = cognilink.map(
    ["./docs/api.pdf", "./docs/security.pdf"],
    model=ChatOpenAI(api_key="sk-..."),
    relationships=[
        {"upstream": "NODE_API", "downstream": "NODE_SECURITY", "type": "constrains"}
    ]
)
```

### Custom summarization prompt

```python
result = cognilink.map(
    "./doc.pdf",
    model=my_model,
    prompt="Summarize focusing on security implications and risks."
)
```

### Using different LLM providers

```python
# Anthropic
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")

# Ollama (local)
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3")

# AWS Bedrock
from langchain_aws import ChatBedrock
model = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")

result = cognilink.map("./doc.pdf", model=model)
```

## CLI Usage

```bash
# OpenAI
python run_cognilink.py --pdf_path ./doc.pdf --model openai --model_id gpt-4o --api_key sk-...

# Ollama (local, no API key needed)
python run_cognilink.py --pdf_path ./doc.pdf --model ollama --model_id llama3

# Multiple files
python run_cognilink.py --pdf_path ./a.pdf ./b.pdf --model openai --model_id gpt-4o

# Persistent mode
python run_cognilink.py --pdf_path ./doc.pdf --model openai --model_id gpt-4o --persist
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  cognilink.map()                      │
├─────────────────────────────────────────────────────┤
│  SessionOrchestrator                                 │
│  ┌──────────┐ ┌──────────┐ ┌───────────────────┐   │
│  │Extractor │ │GraphStore│ │MutationCascader   │   │
│  │Registry  │ │(SQLite)  │ │(BFS + Topo Sort)  │   │
│  └──────────┘ └──────────┘ └───────────────────┘   │
│  ┌──────────────────────┐ ┌─────────────────────┐   │
│  │ LangChain Model      │ │ Visualizer          │   │
│  │ (user-provided)      │ │ (HTML + JSON)       │   │
│  └──────────────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Flow:** Documents → Extract (zero tokens) → Store in graph → Summarize via LLM → Link edges → Render visualization

**Mutation cascade:** Update a node → BFS marks downstream as STALE → Regenerate in topological order

## Advanced Usage

For more control, use the `SessionOrchestrator` directly:

```python
from pathlib import Path
from cognilink import SessionOrchestrator, SessionConfig
from langchain_openai import ChatOpenAI

config = SessionConfig(
    model=ChatOpenAI(api_key="sk-..."),
    workspace_path=Path("./output"),
    persist=True,
    db_path=Path("./output/knowledge.db"),
)

orch = SessionOrchestrator(config)

# Ingest
result = orch.ingest_documents(["./doc1.pdf", "./doc2.txt"])

# Add relationships
orch.add_edge("NODE_DOC1", "NODE_DOC2", "defines_logic_for")

# Update a document (triggers cascade)
cascade = orch.update_node("NODE_DOC1", "New content here...")

# Regenerate stale nodes
orch.regenerate_stale()

# Get context for an agent
context = orch.get_context("NODE_DOC2")

orch.close()
```

## License

MIT
