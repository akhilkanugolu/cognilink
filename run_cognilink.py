#!/usr/bin/env python3
"""CogniLink CLI entry point — ingest documents and build a knowledge graph.

Usage:
    python run_cognilink.py --pdf_path ./doc.pdf --model openai --api_key sk-... --model_id gpt-4o
    python run_cognilink.py --pdf_path ./doc.pdf --model ollama --model_id llama3
    python run_cognilink.py --pdf_path ./a.pdf ./b.pdf --model anthropic --api_key sk-ant-...
"""
import argparse
import sys
from pathlib import Path


def _create_model(provider: str, model_id: str, api_key: str | None = None, api_base: str | None = None):
    """Create a LangChain chat model based on provider string.

    Args:
        provider: One of "openai", "anthropic", "ollama", "bedrock".
        model_id: The model identifier (e.g. "gpt-4o", "llama3", "claude-3-5-sonnet-20241022").
        api_key: Optional API key.
        api_base: Optional API base URL.

    Returns:
        A LangChain BaseChatModel instance.
    """
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            print("Error: Install langchain-openai: pip install langchain-openai")
            sys.exit(1)
        kwargs = {"model": model_id}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["base_url"] = api_base
        return ChatOpenAI(**kwargs)

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            print("Error: Install langchain-anthropic: pip install langchain-anthropic")
            sys.exit(1)
        kwargs = {"model": model_id}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            print("Error: Install langchain-ollama: pip install langchain-ollama")
            sys.exit(1)
        kwargs = {"model": model_id}
        if api_base:
            kwargs["base_url"] = api_base
        return ChatOllama(**kwargs)

    elif provider == "bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            print("Error: Install langchain-aws: pip install langchain-aws")
            sys.exit(1)
        return ChatBedrock(model_id=model_id)

    else:
        print(f"Error: Unknown provider '{provider}'. Use: openai, anthropic, ollama, bedrock")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CogniLink — Build a knowledge graph from your documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cognilink.py --pdf_path ./doc.pdf --model openai --model_id gpt-4o --api_key sk-...
  python run_cognilink.py --pdf_path ./doc.pdf --model ollama --model_id llama3
  python run_cognilink.py --pdf_path ./a.pdf ./b.pdf --model anthropic --model_id claude-3-5-sonnet-20241022
        """,
    )
    parser.add_argument("--pdf_path", type=str, nargs="+", required=True, help="Path(s) to files to ingest")
    parser.add_argument("--model", type=str, required=True, help="LLM provider: openai, anthropic, ollama, bedrock")
    parser.add_argument("--model_id", type=str, default="gpt-4o", help="Model identifier (default: gpt-4o)")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace directory for output (default: current dir)")
    parser.add_argument("--persist", action="store_true", help="Persist graph to disk")
    parser.add_argument("--db_path", type=str, default=None, help="Path to SQLite database file (for persistent mode)")
    parser.add_argument("--api_key", type=str, default=None, help="API key (optional, uses env vars by default)")
    parser.add_argument("--api_base", type=str, default=None, help="API base URL (optional)")

    args = parser.parse_args()

    # Create the LangChain model
    chat_model = _create_model(
        provider=args.model,
        model_id=args.model_id,
        api_key=args.api_key,
        api_base=args.api_base,
    )

    from cognilink.core.models import SessionConfig
    from cognilink.orchestrator import SessionOrchestrator

    workspace = Path(args.workspace)
    db_path = Path(args.db_path) if args.db_path else (workspace / "cognilink.db" if args.persist else None)

    config = SessionConfig(
        model=chat_model,
        workspace_path=workspace,
        persist=args.persist,
        db_path=db_path,
    )

    orchestrator = SessionOrchestrator(config)

    try:
        result = orchestrator.ingest_documents(args.pdf_path)
        print(f"\n✓ Ingested {len(result.node_ids)} documents")
        print(f"  Nodes: {', '.join(result.node_ids)}")
        print(f"  Edges: {result.edges_created}")

        outputs = orchestrator.render_visualization()
        print(f"\n✓ Visualization rendered:")
        print(f"  HTML: {outputs['html']}")
        print(f"  JSON: {outputs['json']}")
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
