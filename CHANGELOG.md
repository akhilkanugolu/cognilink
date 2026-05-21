# Changelog

All notable changes to CogniLink will be documented in this file.

## [0.1.0] - 2024-12-01

### Added
- Initial release of CogniLink
- `cognilink.map()` — simple one-function entry point
- `SessionOrchestrator` — full-featured orchestration API
- `GraphStore` — SQLite-backed graph storage (ephemeral + persistent modes)
- `MutationCascader` — BFS cascade propagation + topological regeneration
- `ExtractorRegistry` — plugin-based concurrent document extraction
- PDF parser via PyPDF2
- Interactive HTML visualization with vis.js
- JSON graph export
- CLI entry point (`run_cognilink.py`)
- LangChain-native LLM integration (any BaseChatModel)
- Node state machine (PENDING → RUNNING → COMPLETED, any → STALE)
- Cycle detection in dependency graphs
- Atomic file writes for visualization outputs
- Comprehensive test suite

### Architecture
- Core dependency: `langchain-core` (user brings their own chat model)
- Template engine: Jinja2
- Storage: SQLite (in-memory or file-based)
- Visualization: vis.js via CDN
