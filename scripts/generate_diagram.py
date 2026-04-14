#!/usr/bin/env python3
"""
Script to generate a Mermaid diagram of the ResearchPal LangGraph.

Usage:
    python scripts/generate_diagram.py

This will output Mermaid markup that can be rendered into a visual diagram.
You can paste the output into:
- https://mermaid.live
- GitHub markdown
- VS Code with Mermaid extension
- Or save as .mmd file and convert to PNG/SVG
"""

import sys
import os

# Permet d'importer les modules src/ depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import build_graph


def main():
    """Generate and print the Mermaid diagram for the LangGraph."""
    print("Generating LangGraph Mermaid diagram...")
    print("=" * 50)

    try:
        # Build graph without checkpointer for cleaner visualization
        # This might take a moment as it initializes LLMs
        print("Building graph (this may take a few seconds)...")
        graph = build_graph(use_checkpointer=False)

        # Generate Mermaid markup
        mermaid_code = graph.get_graph().draw_mermaid()

        print("Mermaid Diagram Code:")
        print("-" * 30)
        print(mermaid_code)
        print("-" * 30)
        print("\nTo visualize:")
        print("1. Copy the code above")
        print("2. Paste into https://mermaid.live")
        print("3. Or save as diagram.mmd and use a Mermaid renderer")
        print("4. Or paste into GitHub markdown (supports Mermaid)")

    except Exception as e:
        print(f"Error generating diagram: {e}")
        print("Make sure all dependencies are installed.")
        print("If this fails due to LLM initialization, try running with Ollama started.")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()