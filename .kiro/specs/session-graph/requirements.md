# Requirements Document

## Introduction

CogniLink is a dual-engine stateful context middleware for multi-agent systems, published as an open-source Python 3.11+ package (`cognilink`) on PyPI under the MIT license. It maintains a directed graph of workspace data blocks (nodes) and their relationships (edges), supporting both ephemeral in-memory sessions and persistent SQLite storage. The system ingests documents (PDF, text, Excel) through a plugin-based extractor registry, summarizes content via any LangChain-compatible chat model (`BaseChatModel` from `langchain-core` — the user brings their own configured model instance supporting OpenAI, Bedrock, Ollama, Anthropic, Azure, Google, and any other LangChain-integrated provider), implements real-time mutation cascade propagation, provides bounded context retrieval for agent consumption, and outputs interactive HTML visualizations (vis.js) alongside JSON exports.

## Glossary

- **CogniLink**: The overall system — a dual-engine stateful context middleware for multi-agent systems
- **ExtractorRegistry**: The plugin-based component responsible for registering file parsers and executing concurrent document extraction
- **GraphStore**: The SQLite-backed storage engine managing nodes and edges, supporting both ephemeral (in-memory) and persistent (file-based) modes
- **LangChain BaseChatModel**: The LLM abstraction — CogniLink accepts any `BaseChatModel` instance from `langchain-core`. The user configures their own model (from `langchain-openai`, `langchain-aws`, `langchain-community`, etc.) and passes it to CogniLink. CogniLink calls `model.invoke()` with the appropriate prompts.
- **MutationCascader**: The component that propagates STALE state to downstream nodes when an upstream node is mutated
- **Visualizer**: The component that renders interactive HTML graphs (vis.js) and JSON exports from graph state
- **SessionOrchestrator**: The top-level coordinator that wires all components together and exposes the public API
- **Node**: An atomic data block in the graph representing an ingested document, with metadata including raw content, summary, state, and source information
- **Edge**: A directed relationship between two nodes, with a semantic relationship type label
- **NodeState**: The lifecycle state of a node — one of PENDING, RUNNING, COMPLETED, or STALE
- **Context_Payload**: The bounded response returned by the context retrieval API, containing target node content and upstream parent summaries
- **Cascade_Report**: The result of a mutation cascade operation, listing all affected downstream node IDs and traversal depth

## Requirements

### Requirement 1: Document Extraction

**User Story:** As a developer integrating CogniLink into a multi-agent pipeline, I want to ingest heterogeneous document formats concurrently, so that I can build a knowledge graph from my workspace without manual preprocessing.

#### Acceptance Criteria

1. WHEN a list of file paths is provided to the ExtractorRegistry, THE ExtractorRegistry SHALL extract raw text from each file concurrently using a ThreadPoolExecutor
2. WHEN a file extension has a registered parser, THE ExtractorRegistry SHALL use that parser to extract text from the file
3. WHEN a file extension has no registered parser, THE ExtractorRegistry SHALL fall back to reading the file as plain UTF-8 text
4. WHEN extraction completes for a file, THE ExtractorRegistry SHALL produce a deterministic node ID derived from the filename in the format NODE_{UPPERCASE_NAME}
5. WHEN the same set of file paths is extracted multiple times, THE ExtractorRegistry SHALL produce identical results each time
6. WHEN a file path points to a non-existent or unreadable file, THE ExtractorRegistry SHALL skip that file, log a warning, and continue processing remaining files
7. THE ExtractorRegistry SHALL consume zero LLM tokens during the extraction phase

### Requirement 2: Graph Storage

**User Story:** As a developer, I want a reliable graph storage engine with dual-lifetime modes, so that I can choose between ephemeral session-scoped graphs and persistent cross-session knowledge bases.

#### Acceptance Criteria

1. WHEN persist is set to False, THE GraphStore SHALL use an in-memory SQLite database that is automatically deleted when the session ends
2. WHEN persist is set to True with a db_path, THE GraphStore SHALL use a file-based SQLite database that survives process termination
3. WHEN a persistent GraphStore is reopened against an existing db_path, THE GraphStore SHALL load all previously committed nodes and edges
4. WHEN a node is inserted, THE GraphStore SHALL store it with all required fields and make it immediately retrievable
5. WHEN a duplicate node ID is inserted, THE GraphStore SHALL raise a NodeAlreadyExistsError and leave the database unchanged
6. WHEN an edge is inserted, THE GraphStore SHALL verify that both upstream_id and downstream_id reference existing nodes
7. WHEN an edge with upstream_id equal to downstream_id is inserted, THE GraphStore SHALL reject it as a self-edge violation
8. WHEN a duplicate edge with the same upstream_id, downstream_id, and relationship_type is inserted, THE GraphStore SHALL reject it
9. WHEN a mutation operation fails, THE GraphStore SHALL roll back the transaction and leave the database in a consistent state
10. WHEN persist is False, THE GraphStore SHALL write no data to disk at any point during the session lifecycle
11. THE GraphStore SHALL enforce that node state values are restricted to PENDING, RUNNING, COMPLETED, or STALE
12. THE GraphStore SHALL maintain indexed lookups on upstream and downstream edge columns for efficient traversal

### Requirement 3: LangChain Model Integration

**User Story:** As a developer, I want to pass any LangChain-compatible chat model to CogniLink, so that I can use any LLM provider (OpenAI, Bedrock, Ollama, Azure, Anthropic, Google, etc.) without CogniLink needing provider-specific code.

#### Acceptance Criteria

1. THE SessionConfig SHALL accept a `model` parameter of type `BaseChatModel` from `langchain-core`
2. WHEN summarization is needed, THE SessionOrchestrator SHALL call `model.invoke([SystemMessage(...), HumanMessage(...)])` on the user-provided model instance
3. WHEN the model invocation returns an AIMessage, THE SessionOrchestrator SHALL extract the text content from `AIMessage.content`
4. THE CogniLink package SHALL NOT include any provider-specific LLM code — all provider routing is handled by the user's LangChain model instance
5. WHEN the user-provided model raises an exception during `model.invoke()`, THE SessionOrchestrator SHALL store the node with state PENDING and empty summary, and raise an LLMUnavailableError
6. THE CogniLink package SHALL provide a top-level `cognilink.map()` convenience function that accepts file path(s), a `BaseChatModel` instance, and optional parameters (prompt, relationships) for the simplest use case

### Requirement 4: Mutation Cascade Propagation

**User Story:** As a developer working with interconnected documents, I want upstream changes to automatically flag downstream dependents as stale, so that I can maintain consistency across my knowledge graph without manual tracking.

#### Acceptance Criteria

1. WHEN a node's content is updated, THE MutationCascader SHALL traverse all transitively downstream nodes via BFS and mark them as STALE
2. WHEN propagate_stale completes, THE MutationCascader SHALL return a CascadeReport containing the source node ID, all affected stale node IDs, and the maximum traversal depth
3. WHEN the dependency graph contains cycles, THE MutationCascader SHALL detect them via a visited set and terminate traversal without infinite loops
4. WHEN propagate_stale is called, THE MutationCascader SHALL leave all nodes that are not transitively downstream of the mutated node unchanged
5. WHEN stale nodes need regeneration, THE MutationCascader SHALL compute a topological sort order such that for any edge (A, B) where both are stale, A appears before B in the regeneration list
6. IF a cycle is detected among stale nodes during topological sort, THEN THE MutationCascader SHALL raise a CycleDetectedError with the list of involved nodes

### Requirement 5: Context Retrieval

**User Story:** As an agent consuming CogniLink context, I want bounded context windows that include relevant upstream dependencies, so that I can make informed decisions without token budget overflow.

#### Acceptance Criteria

1. WHEN get_context is called with a target node ID, THE SessionOrchestrator SHALL return the target node's full content along with summaries of all direct upstream parent nodes
2. WHEN assembling context, THE SessionOrchestrator SHALL include only summaries (not full content) for parent nodes to maintain bounded payload size
3. WHEN assembling context, THE SessionOrchestrator SHALL include the relationship chain (edges) connecting parents to the target node
4. WHEN get_context is called, THE SessionOrchestrator SHALL exclude all nodes that are not the target or direct upstream parents of the target
5. IF get_context is called with a non-existent node ID, THEN THE SessionOrchestrator SHALL raise a NodeNotFoundError

### Requirement 6: Interactive Visualization

**User Story:** As a developer or stakeholder, I want an interactive HTML graph visualization of my knowledge graph, so that I can visually explore document relationships and node states.

#### Acceptance Criteria

1. WHEN render is called, THE Visualizer SHALL produce a self-contained HTML file with embedded vis.js that requires no external CDN dependencies
2. WHEN rendering nodes, THE Visualizer SHALL color-code them by state: green for COMPLETED, yellow for RUNNING, red for STALE, and gray for PENDING
3. WHEN rendering edges, THE Visualizer SHALL display the relationship_type as a label on each edge with directional arrows
4. WHEN a user clicks a node in the HTML visualization, THE Visualizer SHALL reveal node details including summary, source path, and state
5. WHEN render is called, THE Visualizer SHALL also produce a JSON export containing the full graph structure with all nodes and edges
6. WHEN writing output files, THE Visualizer SHALL use atomic writes (write to temp file then rename) to prevent partial file corruption
7. WHEN render completes, THE Visualizer SHALL have produced a visual node for every node in the GraphStore and a visual edge for every edge, with no omissions or phantom entries

### Requirement 7: Session Orchestration

**User Story:** As a developer, I want a single entry point that coordinates document ingestion, mutation handling, context retrieval, and visualization, so that I can use CogniLink through a clean, unified API.

#### Acceptance Criteria

1. WHEN ingest_documents is called with file paths, THE SessionOrchestrator SHALL execute the full pipeline: extract, store, summarize via LLM, create edges, and render visualization
2. WHEN ingest_documents is called with a relationships list, THE SessionOrchestrator SHALL create directed edges between the specified nodes
3. WHEN update_node is called with new content, THE SessionOrchestrator SHALL update the node content, regenerate its summary, and trigger mutation cascade propagation
4. WHEN a file is re-ingested and produces a node ID that already exists, THE SessionOrchestrator SHALL treat it as an update and trigger the mutation cascade path
5. WHEN regenerate_stale is called, THE SessionOrchestrator SHALL regenerate all STALE nodes in topological order and set their state to COMPLETED
6. WHEN close is called, THE SessionOrchestrator SHALL cleanly close the database connection and release all resources
7. IF the visualization output directory is not writable, THEN THE SessionOrchestrator SHALL raise a VisualizationWriteError while keeping the in-memory graph state valid

### Requirement 8: Node State Machine

**User Story:** As a developer, I want predictable node lifecycle states, so that I can reason about which nodes are current, in-progress, or need regeneration.

#### Acceptance Criteria

1. WHEN a node is first created during ingestion, THE GraphStore SHALL set its state to PENDING
2. WHEN processing begins on a node (summarization or regeneration), THE GraphStore SHALL transition its state to RUNNING
3. WHEN processing completes successfully, THE GraphStore SHALL transition the node state to COMPLETED
4. WHEN an upstream node is mutated, THE MutationCascader SHALL transition all downstream nodes to STALE regardless of their current state
5. WHEN a STALE node begins regeneration, THE GraphStore SHALL transition its state to RUNNING
6. THE GraphStore SHALL reject any state transition not in the set: PENDING to RUNNING, RUNNING to COMPLETED, any state to STALE, STALE to RUNNING

### Requirement 9: Error Handling and Recovery

**User Story:** As a developer, I want graceful error handling with clear recovery paths, so that partial failures do not corrupt my graph state.

#### Acceptance Criteria

1. IF a file is not found during extraction, THEN THE ExtractorRegistry SHALL skip it, log a warning, and continue processing remaining files
2. IF the LLM provider is unavailable during ingestion, THEN THE SessionOrchestrator SHALL store nodes with PENDING state and raise LLMUnavailableError with retry guidance
3. IF update_node fails mid-operation due to an LLM timeout, THEN THE SessionOrchestrator SHALL retain the node's previous state and not trigger any cascade
4. IF a persistent SQLite database file is corrupted or has a schema mismatch, THEN THE GraphStore SHALL raise a DatabaseCorruptionError on initialization without silently overwriting the file
5. IF the user-provided BaseChatModel raises an exception during model.invoke(), THEN THE SessionOrchestrator SHALL catch the exception and raise LLMUnavailableError with the original error details

### Requirement 10: Packaging and Distribution

**User Story:** As a Python developer, I want to install CogniLink from PyPI with minimal dependencies and optional extras, so that I can integrate it into my project without bloating my dependency tree.

#### Acceptance Criteria

1. THE CogniLink package SHALL be installable via pip as `cognilink` with Python 3.11 or higher
2. THE CogniLink package SHALL have only `langchain-core` and `jinja2` as core dependencies
3. WHERE PDF extraction is needed, THE CogniLink package SHALL provide a `pdf` optional extra installing PyPDF2
4. WHERE Excel extraction is needed, THE CogniLink package SHALL provide an `excel` optional extra installing openpyxl
5. THE CogniLink package SHALL use a top-level package layout with `cognilink/` for simplicity and contributor accessibility
6. THE CogniLink package SHALL be licensed under the MIT license

### Requirement 11: Security and Data Safety

**User Story:** As a developer handling sensitive documents, I want CogniLink to protect my data and credentials, so that I can use it safely in production environments.

#### Acceptance Criteria

1. WHEN persist is False, THE GraphStore SHALL ensure no data is written to disk at any point during the session
2. WHEN persist is True, THE GraphStore SHALL set SQLite file permissions to user-only access (0600)
3. THE CogniLink package SHALL use parameterized SQL queries exclusively, never interpolating raw content into SQL strings
4. THE Visualizer SHALL include only node summaries in HTML and JSON outputs, never full raw content, to limit data leakage
5. THE CogniLink package SHALL not log, persist, or include LLM API keys in any output files
6. THE CogniLink package SHALL not transmit telemetry or usage data to any external service

### Requirement 12: Section-Level PDF Parsing (Optional — PageIndex-style)

**User Story:** As a developer working with large, structured PDF documents, I want the option to parse PDFs into hierarchical section nodes rather than treating the entire document as a single node, so that I can query and link individual sections across documents with fine-grained context retrieval.

#### Acceptance Criteria

1. WHEN `section_mode=True` is set in SessionConfig, THE SessionOrchestrator SHALL parse PDF documents into a hierarchical tree of section nodes using the SectionParser, with each section becoming a child node linked to a parent document root node via a "contains_section" edge
2. WHEN section-level parsing is active, THE SectionParser SHALL detect headings and sections in the PDF via font size analysis or Table of Contents extraction, and create a recursive tree of sections and sub-sections
3. WHEN a section node is created, THE Node SHALL include `page_start` and `page_end` fields indicating the page range of that section within the source PDF
4. WHEN a parent section node is mutated (content updated), THE MutationCascader SHALL cascade STALE state to all child section nodes via the existing "contains_section" edges, consistent with CogniLink's standard cascade behavior
5. WHEN `section_mode=False` (the default), THE SessionOrchestrator SHALL treat the entire PDF as a single node with no section decomposition, preserving the existing behavior unchanged
6. WHEN section-level parsing is active, THE SectionParser SHALL generate an LLM summary for each individual section node
7. WHEN section-level parsing is active, cross-document section linking SHALL be supported — a section node in one PDF can be linked to a section node in another PDF via standard edge creation (add_edge)
8. WHEN section-level parsing is active, THE Node source_type for section nodes SHALL be "SECTION", distinguishing them from top-level "PDF" document root nodes
