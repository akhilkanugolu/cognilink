"""CogniLink basic usage example.

Demonstrates the simple cognilink.map() API with different LLM providers.

Prerequisites:
    pip install cognilink[pdf] langchain-openai
    # or: pip install cognilink[pdf] langchain-ollama (for local models)
"""

import cognilink


def example_openai():
    """Use OpenAI as the LLM provider."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o", api_key="sk-...")

    result = cognilink.map(
        "./docs/spec.pdf",
        model=model,
        prompt="Summarize this document focusing on key technical decisions.",
    )

    print("Nodes:", len(result["nodes"]))
    for node in result["nodes"]:
        print(f"  {node['id']}: {node['summary'][:80]}...")

    print(f"\nHTML: {result['html_path']}")
    print(f"JSON: {result['json_path']}")


def example_ollama():
    """Use a local Ollama model (no API key needed)."""
    from langchain_ollama import ChatOllama

    model = ChatOllama(model="llama3")

    result = cognilink.map("./docs/readme.md", model=model)
    print("Nodes:", result["nodes"])


def example_multiple_docs():
    """Ingest multiple documents with relationships."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o", api_key="sk-...")

    result = cognilink.map(
        ["./docs/api_spec.pdf", "./docs/security_policy.pdf"],
        model=model,
        relationships=[
            {
                "upstream": "NODE_API_SPEC",
                "downstream": "NODE_SECURITY_POLICY",
                "type": "must_comply_with",
            }
        ],
    )

    print(f"Created {len(result['nodes'])} nodes and {len(result['edges'])} edges")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    # example_openai()
    # example_ollama()
    # example_multiple_docs()
    print("Edit this file and uncomment an example to run it.")
