"""Command-line interface for asking the bounded multi-agent system a question."""
from __future__ import annotations

import argparse
import sys
from typing import Any

from .orchestrator import final_answer_text, run_agent_query


def _items(result: dict[str, Any], category: str) -> list[dict[str, Any]]:
    analysis = result.get("analysis")
    if isinstance(analysis, dict) and isinstance(analysis.get(category), list):
        return [item for item in analysis[category] if isinstance(item, dict)]
    return []


def _sources(result: dict[str, Any]) -> list[Any]:
    sources = result.get("sources")
    if isinstance(sources, list) and sources:
        return sources
    gathered = []
    seen = set()
    for agent in result.get("agent_results", []):
        if not isinstance(agent, dict):
            continue
        for source in agent.get("sources", []) if isinstance(agent.get("sources"), list) else []:
            key = (source.get("source"), source.get("page"), source.get("section")) if isinstance(source, dict) else str(source)
            if key not in seen:
                seen.add(key)
                gathered.append(source)
    return gathered


def _source_label(source: Any) -> str:
    if isinstance(source, str):
        return source
    if not isinstance(source, dict):
        return str(source)
    label = source.get("source") or source.get("title") or "Unknown source"
    details = []
    if source.get("page"):
        details.append(f"page {source['page']}")
    if source.get("section"):
        details.append(f"section {source['section']}")
    return f"{label} ({', '.join(details)})" if details else str(label)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Ask the Argos financial intelligence agents a question.")
    parser.add_argument("question", nargs="+", help="Question for the multi-agent financial intelligence system")
    args = parser.parse_args()
    result = run_agent_query(" ".join(args.question))

    print(f"Question: {result.get('question') or ' '.join(args.question)}")
    print(f"Agents: {', '.join(result.get('selected_agents') or []) or 'none'}")
    print(f"Status: {result.get('status') or 'unknown'}")
    print()
    print("=" * 50)
    print("FINOP AI ANSWER")
    print("=" * 50)
    answer = final_answer_text(result)
    print(answer if answer else "No natural-language answer was returned by the selected agents.")

    recommendations = _items(result, "recommendations")
    if recommendations:
        print("\nRecommendations:")
        for item in recommendations:
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                print(f"- {text.strip()}")

    sources = _sources(result)
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {_source_label(source)}")

    limitations = result.get("limitations")
    if isinstance(limitations, list) and limitations:
        print("\nLimitations:")
        for limitation in limitations:
            if str(limitation).strip():
                print(f"- {limitation}")

    print("\nReports:")
    print("output/agentic_financial_analysis.json")
    print("output/agentic_financial_analysis.md")


if __name__ == "__main__":
    main()
