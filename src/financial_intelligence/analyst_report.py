"""Serialization helpers for Phase 4 analyst reports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .prompt_builder import SECTIONS


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# LLM Financial Analysis", "", f"**Status:** {report.get('status', 'unknown')}",
             f"**Provider / model:** {report.get('provider', 'n/a')} / {report.get('model', 'n/a')}", ""]
    for section in SECTIONS:
        lines.extend(["## " + section.replace("_", " ").title(), ""])
        items = report.get(section, [])
        if not items:
            lines.extend(["_No evidence-backed items available._", ""])
        for item in items:
            refs = ", ".join(item.get("evidence_ids", []))
            lines.append(f"- **{item.get('classification', 'fact').title()}:** {item.get('text', '')}" + (f" _(Evidence: {refs})_" if refs else ""))
        lines.append("")
    if report.get("error"):
        lines.extend(["## Generation status", "", report["error"], ""])
    return "\n".join(lines)


def write_reports(report: dict[str, Any], output_dir: str | Path = "output") -> tuple[Path, Path]:
    """Write JSON and Markdown files and return their paths."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "llm_financial_analysis.json", out / "llm_financial_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
