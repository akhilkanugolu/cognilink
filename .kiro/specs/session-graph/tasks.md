# Implementation Plan: CogniLink (session-graph)

## Overview

Implement CogniLink as a Python 3.11+ open-source package with a hybrid layout (`cognilink/` at top level, `run_cognilink.py` entry point). The implementation proceeds bottom-up: scaffolding → data models → graph store → extractors → LangChain model integration → mutation cascader → visualizer → orchestrator → entry point → documentation. Each layer builds on the previous, with property-based and unit tests validating correctness at each stage.

## Tasks

- [x] 1. Project scaffolding and package structure
  - [x] 1.1 Create pyproject.toml with build system, dependencies, and optional extras
    - Configure hatchling build backend
    - Define core dependencies (langchain-core, jinja2) and optional extras (pdf, excel, dev)
    - Set Python >=3.11 requirement, MIT license, classifiers
    - Include tool configs for pytest, ruff, and mypy
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 1.2 Create package directory structure and __init__.py files
    - Create cognilink/, cognilink/core/, cognilink/extract/, cognilink/viz/, cognilink/viz/templates/
    - Create tests/, tests/unit/, tests/property/, tests/integration/
    - Create examples/ directory
    - Add __init__.py with public API exports in cognilink/__init__.py
    - _Requirements: 10.6_

  - [x] 1.3 Create CI/CD workflow files
    - Create .github/workflows/ci.yml with Python 3.11/3.12/3.13 matrix
    - Include linting (ruff), type checking (mypy), and test steps
    - Create .github/workflows/release.yml for PyPI publishing on tag
    - _Requirements: 10.1_

  - [x] 1.4 Create supporting project files
    - Create requirements.txt with pinned dev dependencies
    - Create .gitignore for Python projects
    - Create .env.example documenting required environment variables (LLM API keys, model IDs)
    - Create LICENSE file (MIT)
    - _Requirements: 10.7, 11.5_

- [x] 2. Core data models and exceptions
  - [x] 2.1 Implement data models in cognilink/core/models.py
    - Define NodeState enum (PENDING, RUNNING, COMPLETED, STALE)
    - Define Node dataclass with all fields (id, source_type, source_path, raw_content, summary, system_role_prompt, output_artifact, state)
    - Define Edge dataclass (id, upstream_id, downstream_id, relationship_type)
    - Define SessionConfig (with model: BaseChatModel), IngestionResult, ContextPayload, CascadeReport dataclasses
    - _Requirements: 2.11, 8.1, 8.6_

  - [x] 2.2 Implement custom exceptions in cognilink/core/exceptions.py
    - Define NodeAlreadyExistsError, NodeNotFoundError, CycleDetectedError
    - Define LLMUnavailableError, DatabaseCorruptionError, VisualizationWriteError
    - Define InvalidStateTransitionError
    - _Requirements: 2.5, 4.6, 5.5, 7.7, 9.1, 9.2, 9.4_

  - [ ]* 2.3 Write property test for state machine validity
    - **Property 13: State Machine Validity**
    - **Validates: Requirements 8.1, 8.6, 2.11**

- [x] 3. Graph storage engine
  - [x] 3.1 Implement GraphStore in cognilink/core/graph_store.py
    - Initialize SQLite with schema (nodes table, edges table, indices)
    - Support dual-mode: in-memory (:memory:) when persist=False, file-based when persist=True
    - Implement insert_node with duplicate detection (raise NodeAlreadyExistsError)
    - Implement get_node, update_node, delete_node, get_all_nodes
    - Implement insert_edge with referential integrity checks (both nodes exist, no self-edges, no duplicates)
    - Implement get_downstream, get_upstream, delete_edge, get_all_edges
    - Implement set_node_state with state transition validation
    - Wrap all mutations in transactions for atomicity
    - Create indices on upstream_id and downstream_id columns
    - Use parameterized SQL queries exclusively
    - Set file permissions to 0600 for persistent mode
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 11.1, 11.2, 11.3_

  - [ ]* 3.2 Write property test for graph referential integrity
    - **Property 1: Graph Referential Integrity**
    - **Validates: Requirements 2.6, 2.7, 2.8**

  - [ ]* 3.3 Write property test for node insert-retrieve round trip
    - **Property 2: Node Insert-Retrieve Round Trip**
    - **Validates: Requirements 2.4, 2.5**

  - [ ]* 3.4 Write property test for persistence round trip
    - **Property 3: Persistence Round Trip**
    - **Validates: Requirements 2.2, 2.3**

  - [ ]* 3.5 Write unit tests for GraphStore
    - Test CRUD operations for nodes and edges
    - Test state transition enforcement (valid and invalid transitions)
    - Test ephemeral mode writes nothing to disk
    - Test persistent mode survives close/reopen
    - Test transaction rollback on failure
    - Test duplicate node/edge rejection
    - Test self-edge rejection
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11_

- [x] 4. Checkpoint - Core data layer
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Extractor registry and PDF parser
  - [x] 5.1 Implement ExtractorRegistry in cognilink/extract/registry.py
    - Implement register_parser(extension, parser_fn) for plugin registration
    - Implement parse_workspace_concurrently(paths) using ThreadPoolExecutor
    - Generate deterministic node IDs: NODE_{UPPERCASE_FILENAME_WITHOUT_EXT}
    - Implement fallback plain UTF-8 reader for unregistered extensions
    - Skip non-existent/unreadable files with warning log, continue processing
    - Ensure zero LLM token consumption during extraction
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 5.2 Implement PDF parser in cognilink/extract/pdf.py
    - Create pdf_parser function using PyPDF2 to extract text from PDF files
    - Handle multi-page PDFs by concatenating page text
    - Gracefully handle encrypted or malformed PDFs
    - _Requirements: 1.2_

  - [ ]* 5.3 Write property test for idempotent extraction
    - **Property 9: Idempotent Extraction**
    - **Validates: Requirements 1.4, 1.5**

  - [ ]* 5.4 Write property test for extraction parser dispatch
    - **Property 10: Extraction Parser Dispatch**
    - **Validates: Requirements 1.2, 1.3, 1.6**

  - [ ]* 5.5 Write unit tests for ExtractorRegistry
    - Test parser registration and dispatch
    - Test concurrent extraction with multiple files
    - Test fallback to UTF-8 for unknown extensions
    - Test deterministic node ID generation
    - Test graceful handling of missing files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 5.6 Implement SectionParser in cognilink/extract/section_parser.py
    - Implement heading detection via font size analysis (using PyPDF2 page-level metadata)
    - Implement Table of Contents extraction as fallback heading source
    - Build recursive section tree from detected headings (SectionInfo dataclass)
    - Implement create_section_nodes() to insert section nodes with parent_node_id, page_start, page_end
    - Create "contains_section" edges from parent to child sections
    - Generate per-section LLM summaries via the user-provided BaseChatModel
    - Integrate with SessionOrchestrator: when section_mode=True, invoke SectionParser after PDF extraction
    - Ensure section nodes use source_type="SECTION"
    - _Requirements: 12.1, 12.2, 12.3, 12.6, 12.7, 12.8_

  - [ ]* 5.7 Write unit tests for SectionParser
    - Test heading detection from a sample PDF with varying font sizes
    - Test recursive tree building from flat heading list
    - Test section node creation with correct parent_node_id and page ranges
    - Test "contains_section" edge creation between parent and child sections
    - Test that mutation of a parent section cascades STALE to child sections
    - Test that section_mode=False leaves PDF as a single node (no section decomposition)
    - Test cross-document section linking via add_edge
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

- [x] 6. LangChain model integration
  - [x] 6.1 Integrate LangChain BaseChatModel in orchestrator
    - Add `langchain-core>=0.3.0` as core dependency in pyproject.toml (replace litellm)
    - Implement `_invoke_model(model, system_prompt, user_prompt)` helper in cognilink/orchestrator.py
    - Use `model.invoke([SystemMessage(...), HumanMessage(...)])` for all LLM calls
    - Extract response text from `AIMessage.content`
    - Handle LangChain exceptions (timeouts, auth errors, rate limits) gracefully
    - Wrap exceptions as LLMUnavailableError with original error details
    - Remove `cognilink/inference/` directory entirely (base.py, openai_compat.py, bedrock.py, litellm_provider.py, __init__.py)
    - Remove GenerationResult dataclass from models.py (use AIMessage.content directly)
    - Update SessionConfig to accept `model: BaseChatModel` instead of model string
    - Remove `api_key` and `api_base` fields from SessionConfig (user configures these on their model)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 6.2 Implement cognilink.map() convenience function
    - Create top-level `cognilink.map()` function in cognilink/__init__.py
    - Accept `paths: Union[str, List[str]]` as first argument
    - Accept `model: BaseChatModel` parameter (required)
    - Accept optional `prompt: str` parameter (default: "Summarize the following document in 2-3 sentences.")
    - Accept optional `persist: bool` parameter (default: False)
    - Accept optional `section_mode: bool` parameter (default: False)
    - Accept optional `relationships: List[Dict]` parameter for edge creation
    - Return `MapResult` dataclass with `.nodes`, `.edges`, `.html_path`, `.json_path`
    - Internally create a SessionOrchestrator, ingest documents, collect results, and clean up
    - _Requirements: 3.6_

  - [ ]* 6.3 Write unit tests for LangChain integration
    - Test that orchestrator calls model.invoke() with correct SystemMessage and HumanMessage
    - Test AIMessage.content extraction from mock model response
    - Test error handling when model.invoke() raises exceptions (wraps as LLMUnavailableError)
    - Test cognilink.map() convenience function with mock BaseChatModel
    - Test that no provider-specific code exists in CogniLink
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 7. Mutation cascader
  - [x] 7.1 Implement MutationCascader in cognilink/core/cascader.py
    - Implement propagate_stale(mutated_node_id) using BFS traversal
    - Use visited set for cycle detection to prevent infinite loops
    - Mark all transitively downstream nodes as STALE
    - Return CascadeReport with source_node_id, stale_node_ids, depth_reached
    - Implement get_regeneration_order(stale_ids) using Kahn's topological sort
    - Raise CycleDetectedError if cycle detected among stale nodes during topo sort
    - Implement regenerate_node(node_id, model) for single-node regeneration using BaseChatModel
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.4, 8.5_

  - [ ]* 7.2 Write property test for cascade completeness
    - **Property 4: Cascade Completeness**
    - **Validates: Requirements 4.1, 4.2, 8.4**

  - [ ]* 7.3 Write property test for cascade isolation
    - **Property 5: Cascade Isolation**
    - **Validates: Requirements 4.4**

  - [ ]* 7.4 Write property test for cascade cycle termination
    - **Property 6: Cascade Cycle Termination**
    - **Validates: Requirements 4.3**

  - [ ]* 7.5 Write property test for topological order validity
    - **Property 7: Topological Order Validity**
    - **Validates: Requirements 4.5, 4.6**

  - [ ]* 7.6 Write unit tests for MutationCascader
    - Test linear chain propagation
    - Test branching graph propagation
    - Test diamond dependency propagation
    - Test cycle detection and termination
    - Test topological sort with valid DAG
    - Test CycleDetectedError on cyclic stale subgraph
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 8. Checkpoint - Core engine complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Visualizer (HTML + JSON)
  - [x] 9.1 Create Jinja2 HTML template in cognilink/viz/templates/graph.html
    - Embed vis.js library directly in the template (no external CDN)
    - Include node click handler to reveal details (summary, source_path, state)
    - Style nodes with color-coding by state
    - Display edge labels with directional arrows
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 9.2 Implement Visualizer in cognilink/viz/html_renderer.py and cognilink/viz/json_export.py
    - Implement render_html() using Jinja2 template rendering
    - Color-code nodes: green=COMPLETED, yellow=RUNNING, red=STALE, gray=PENDING
    - Display relationship_type as edge labels with directional arrows
    - Implement export_json() with full graph structure (nodes + edges)
    - Include only summaries in outputs, never raw_content
    - Use atomic writes (write to temp file, then rename) for both HTML and JSON
    - Implement render() that produces both outputs in a single call
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 11.4_

  - [ ]* 9.3 Write property test for visualization fidelity
    - **Property 11: Visualization Fidelity**
    - **Validates: Requirements 6.5, 6.7**

  - [ ]* 9.4 Write property test for visualization correctness
    - **Property 12: Visualization Correctness**
    - **Validates: Requirements 6.2, 6.3**

  - [ ]* 9.5 Write property test for output data minimization
    - **Property 15: Output Data Minimization**
    - **Validates: Requirements 11.4**

  - [ ]* 9.6 Write unit tests for Visualizer
    - Test HTML output is self-contained (no external CDN references)
    - Test node color mapping for each state
    - Test edge label rendering
    - Test JSON export structure matches expected schema
    - Test atomic write behavior (temp file + rename)
    - Test that raw_content is excluded from outputs
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 11.4_

- [x] 10. Session orchestrator
  - [x] 10.1 Implement SessionOrchestrator in cognilink/orchestrator.py
    - Wire together ExtractorRegistry, GraphStore, user-provided BaseChatModel, MutationCascader, Visualizer
    - Implement ingest_documents(paths, relationships) — full pipeline: extract → store → summarize → link → render
    - Handle re-ingestion of existing node IDs as updates (trigger mutation cascade)
    - Implement update_node(node_id, new_content) — update content, regenerate summary, trigger cascade
    - Implement add_edge(upstream_id, downstream_id, relationship_type)
    - Implement get_context(target_node_id) — bounded context with parent summaries
    - Implement regenerate_stale() — regenerate all STALE nodes in topological order
    - Implement render_visualization() and get_graph_state()
    - Implement close() for clean resource release
    - Handle LLM unavailability: store nodes as PENDING, raise LLMUnavailableError
    - Handle visualization write failures: raise VisualizationWriteError, keep graph state valid
    - Retain previous node state if update_node fails mid-operation (no cascade triggered)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 5.1, 5.2, 5.3, 5.4, 5.5, 9.2, 9.3_

  - [ ]* 10.2 Write property test for context boundedness
    - **Property 8: Context Boundedness**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

  - [ ]* 10.3 Write property test for atomic mutations
    - **Property 14: Atomic Mutations**
    - **Validates: Requirements 9.3**

  - [ ]* 10.4 Write property test for regeneration completeness
    - **Property 16: Regeneration Completeness**
    - **Validates: Requirements 7.5**

  - [ ]* 10.5 Write unit tests for SessionOrchestrator
    - Test full ingestion pipeline with mock BaseChatModel
    - Test update_node triggers cascade
    - Test re-ingestion of existing node triggers update path
    - Test get_context returns bounded payload
    - Test regenerate_stale processes nodes in topological order
    - Test LLM unavailability stores PENDING nodes
    - Test visualization write failure keeps graph valid
    - Test close() releases resources
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 11. Checkpoint - Full system integration
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Entry point script and integration tests
  - [x] 12.1 Implement run_cognilink.py entry point
    - Parse CLI arguments: --pdf_path, --workspace, --persist, --db_path, --model_provider, --model_name
    - Dynamically import the user's LangChain model class based on --model_provider
    - Wire up SessionOrchestrator with the user-provided BaseChatModel instance
    - Register default parsers (PDF, text)
    - Execute ingestion and render visualization
    - Print summary of ingested nodes and output paths
    - _Requirements: 7.1, 10.5_

  - [ ]* 12.2 Write integration tests for end-to-end flows
    - Test full ingestion with mock BaseChatModel and sample text files
    - Test mutation cascade integration (ingest → link → mutate → verify stale → regenerate)
    - Test visualization round-trip (render → parse JSON → verify counts)
    - Test persistence lifecycle (create → persist → close → reopen → verify)
    - _Requirements: 7.1, 7.3, 7.5, 2.2, 2.3, 6.5_

- [x] 13. Documentation
  - [x] 13.1 Create README.md with project overview, installation, and quick start
    - Include badges (CI, coverage, PyPI version, license)
    - Document installation with pip (core and optional extras)
    - Provide quick start code example
    - Document CLI usage via run_cognilink.py
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 13.2 Create example scripts and contributing guide
    - Create examples/basic_usage.py (PDF ingestion + visualization using cognilink.map())
    - Create examples/with_bedrock.py (AWS Bedrock via langchain-aws example)
    - Create examples/persist_and_reload.py (persistent mode demo)
    - Create CONTRIBUTING.md with development setup instructions
    - Create CHANGELOG.md with initial version entry
    - _Requirements: 10.6, 10.7_

- [ ] 14. Final checkpoint - All tests pass and documentation complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (17 properties total)
- Unit tests validate specific examples and edge cases
- v1 is PDF-only for extraction; Excel parser structure exists but is not the focus
- The design uses Python directly, so all implementation tasks use Python 3.11+
- All SQL queries must use parameterized statements (never string interpolation)
- Visualization outputs must contain summaries only, never raw_content
- Tasks 5.6 and 5.7 implement the optional PageIndex-style section-level parsing; v1 can ship without them
- CogniLink uses `langchain-core` as its LLM abstraction — the user brings their own configured LangChain model (from langchain-openai, langchain-aws, langchain-community, etc.)
- The `cognilink/inference/` directory contains only a thin `wrapper.py` around the user-provided BaseChatModel — no custom providers

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "3.5", "5.1", "5.2"] },
    { "id": 4, "tasks": ["5.3", "5.4", "5.5", "5.6", "5.7", "6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.4", "7.5", "7.6", "9.1"] },
    { "id": 7, "tasks": ["9.2"] },
    { "id": 8, "tasks": ["9.3", "9.4", "9.5", "9.6", "10.1"] },
    { "id": 9, "tasks": ["10.2", "10.3", "10.4", "10.5"] },
    { "id": 10, "tasks": ["12.1", "12.2"] },
    { "id": 11, "tasks": ["13.1", "13.2"] }
  ]
}
```
