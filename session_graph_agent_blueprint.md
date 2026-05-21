# 📋 Product Requirement Document (PRD) & Open-Source Technical Specification
## Project Name: SessionGraph (The Dual-Engine Stateful Context Middleware)
**Target Consumer:** Multi-Agent Systems, Test Automation Pipelines, and Code-Generation Frameworks
**Compliance Target:** Python 3.11+ Open-Source Packaged Core (Ready for GitHub/GitLab CI Distribution)

---

## 1. Executive Summary & Comparative Landscape

### 1.1 Detailed Problem Statement
Traditional context injection models degrade severely during long developer-agent chat sessions:
1. **Flat Vector RAG Bloat:** Slicing spreadsheets, architecture PDFs, and Jira ticket links into arbitrary flat chunks ruins dependency tracking. If a core requirement falls into block #1 but its structural input constraints land in block #2, vector lookups lose the contextual parent-child hierarchy completely.
2. **Static GraphRAG Rigidity:** Modern open-source repository indexes (like `graphify` or `codegraph`) act as offline, read-only compilers. They scan static repositories, write permanent index directories to disk, and exit. If a running agent changes a functional spec or writes a new test script mid-conversation, the graph breaks because it cannot mutate state variables interactively within the live thread session.

### 1.2 Prior Art Competitive Evaluation
`SessionGraph` targets an unaddressed architectural space right next to current viral tools:
* **`tinyhumansai/openhuman` (Memory Tree vs. Runtime Generation Flow):** While `openhuman` uses local SQLite frameworks to aggregate a passive diary tracker of personal data, `SessionGraph` strips away desktop application wrappers to provide an active developer SDK pipeline.
* **`safishamsi/graphify` (Static Folder Compiler vs. Live-Drift Mutation Scratchpad):** `graphify` proves that building structural graphs yields a massive token reduction. However, it treats graphs as read-only assets on disk. `SessionGraph` runs entirely in volatile computer RAM (`:memory:`), allowing agents to modify data blocks via interactive loops.
* **`colbymchenry/codegraph` (AST Code Syntax vs. Abstract Document Custom Registries):** `codegraph` relies on Tree-sitter to index code languages syntax (`.py`, `.js`). It cannot parse unstructured business sheets or scrape data fields from custom web APIs.

---

## 2. Multi-Source Task Handling Scenario (e.g., 2 PDFs + 1 Jira Ticket Intake)

To implement this tool successfully, your code must handle the simultaneous ingestion of mixed-format inputs without context mixing or leakage. Here is the operational trace for processing **two distinct specification PDFs and one functional Jira Story ticket**:

```text
[Raw Workspace Inputs] ──> 2x PDF Specifications + 1x Jira API JSON Payload
                                     │
                                     ▼
[Phase 1: Ingestion Pass] ───> Programmatic Extractor reads strings (0 tokens consumed)
                                     │
                                     ▼
[Phase 2: Cleanup Pass] ────> Local Micro-Model (3B) executes text-cleansing self-loops
                                     │
                                     ▼
[Phase 3: Network Binding] ──> Graph Schema links cross-platform parallel multi-edges:
                               * PDF_1 ===(defines_security_bounds_for)===> JIRA_TICKET
                               * PDF_2 ===(imposes_ui_layout_rules_on)=====> JIRA_TICKET
```

---

## 3. The Dual-Graph Output Specification

The implementation must establish two parallel, synchronized representations of the active session graph:

### 3.1 Graph Engine #1: The Human-Readable Visual Ledger (`SESSION_MAP.md`)
* **Purpose:** Provides absolute verification transparency for the human user, ensuring the agent maps out logical connections accurately before starting heavy generation tasks.
* **Format:** Formatted Markdown ledger automatically exported to the user's workspace project path.
* **Requirements:**
  * Must output a detailed block list charting every entity node, its tracking type, and its short micro-model summary.
  * Must print a clear visual ASCII/Text tree detailing matching relationships across systems (e.g., `NODE_PDF_1` ===(**restricts_network_api**)===> `NODE_JIRA_STORY`).
  * If the user discovers a missing relationship link, they can modify this text file or pass a query command to patch the memory database variables.

### 3.2 Graph Engine #2: The Agent Context Tunnel (In-Memory SQLite)
* **Purpose:** Protects premium cloud models (Claude 3.5 Sonnet, GPT-4o) from token-window saturation and hallucinations.
* **Format:** Bounded prompt strings fetched dynamically via your custom Python database query methods.
* **Requirements:**
  * Must extract *only* the specific target node content chunk required for the active processing turn.
  * Must trace parent edge tables to append the summaries and outputs of direct upstream dependencies while filtering out all other unrelated files in the workspace.

---

## 4. In-Memory Relational Database Schema Specification

All structural session tracking states must live in a transient, in-memory SQLite database instance to ensure ultra-low processing latency and auto-deletion on session termination.

```sql
-- Main Table for Atomic Workspace Data Blocks
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,         -- 'JIRA', 'PDF', 'EXCEL', 'CUSTOM'
    source_path TEXT NOT NULL,         -- Reference URI or tracking hash
    raw_content TEXT NOT NULL,         -- Text extracted programmatically (0 tokens)
    summary TEXT NOT NULL,             -- 2-3 sentence summary block from local 3B model
    system_role_prompt TEXT NOT NULL,  -- Default personality instructions assigned by type
    output_artifact TEXT,              -- Generated documentation/code logged by steps
    state TEXT CHECK(state IN ('PENDING', 'RUNNING', 'COMPLETED', 'STALE')) DEFAULT 'PENDING'
);

-- Main Table for Directed Cross-Platform Parallel Multi-Edges
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upstream_id TEXT NOT NULL,
    downstream_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,   -- 'defines_logic_for', 'must_comply_with_rules_in'
    FOREIGN KEY(upstream_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(downstream_id) REFERENCES nodes(id) ON DELETE CASCADE
);

-- Performance Optimization Indices
CREATE INDEX IF NOT EXISTS idx_edges_downstream ON edges(downstream_id);
CREATE INDEX IF NOT EXISTS idx_edges_upstream ON edges(upstream_id);
```

---

## 5. Decoupled Core Python SDK Interface

The repository must follow a decoupled layout, separating data storage mechanics from inference wrappers so developers can seamlessly import it as a standard package.

### 5.1 The Extractor Plugin Registry (`extractor.py`)
```python
import os
import concurrent.futures
from typing import Dict, Callable, Any, List

class ExtractorRegistry:
    def __init__(self):
        self._parsers: Dict[str, Callable[[str], str]] = {}

    def register_parser(self, extension: str, parser_fn: Callable[[str], str]):
        """Allows developers to inject custom file extractors cleanly."""
        self._parsers[extension.lower().strip()] = parser_fn

    def parse_workspace_concurrently(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Executes multi-threaded extraction loops across input lists (0 tokens)."""
        def _execute_read(path: str) -> Dict[str, Any]:
            ext = os.path.splitext(path)[1].lower()
            parser = self._parsers.get(ext, lambda p: open(p, 'r', encoding='utf-8').read())
            return {
                "id": f"NODE_{os.path.basename(path).split('.')[0].upper().replace('-', '_')}",
                "ext": ext,
                "raw_text": parser(path)
            }
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(_execute_read, paths))
```

### 5.2 The Unified Inference Interface (`models.py`)
```python
import abc
from typing import Dict, Any, Optional

class LLMProviderInterface(abc.ABC):
    @abc.abstractmethod
    def invoke_generation(self, system_prompt: str, user_prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Standardized wrapper supporting OpenAI, Anthropic, and local Ollama configurations."""
        pass
```

---

## 6. Real-Time Mutation & Downstream Cascade Mechanics

If a requirement changes mid-conversation (e.g., the user updates a lock-out threshold inside their Jira node text), the system prevents structural drift via an automated schema cascade:

1. **Self-Loop Intercept:** The system updates the targeted node's text content inside the SQLite schema while keeping its structural relationship edges completely intact.
2. **Recursive Downstream Traversal:** The mutation script traces the database paths to locate any child components connected to the modified parent node.
3. **Stale Flag Activation:** The system instantly updates the state flag of all affected child nodes to `STATE = 'STALE'`.
4. **Selective Re-Generation:** On the next conversational turn, the orchestrator alerts the user, halts further execution, and forces the model to selectively regenerate *only* the out-of-sync components while preserving the rest of the workspace.

---

## 7. Open-Source Packaging & Pipeline Configurations

To ensure compliance with GitLab open-source standards, provide a clean packaging file and a standard automation runtime configuration mapping testing pipelines.

### 7.1 Package Distribution Manifest (`setup.py`)
```python
from setuptools import setup, find_packages

setup(
    name="session-graph",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
    ],
    python_requires=">=3.11",
)
```

### 7.2 GitLab Automation Verification Pipeline (`.gitlab-ci.yml`)
```yaml
stages:
  - test

run_unit_tests:
  stage: test
  image: python:3.11-slim
  script:
    - pip install --upgrade pip
    - pip install .
    - python -m unittest discover -s tests
```

---

## 8. Explicit Code Generation Guidelines for Agent Sub-Teams

When passing this documentation layout to an execution agent (such as Claude Code or Cursor) to construct your repository files, append this instructional instruction block:

```text
INSTRUCTION FOR CODE GENERATION AGENT:
Act as a principal systems engineer. Implement the above specifications as a highly robust, clean, and type-annotated Python 3.11 library. 
Follow defensive programming practices: wrap all database connections inside secure transaction blocks, include robust exception handling patterns for missing file paths or parsing errors, and separate data structures cleanly from network logic layers. 
Do not output conversational chat explanations or placeholders. Write ready-to-use, complete production scripts matching this requirements blueprint.
```