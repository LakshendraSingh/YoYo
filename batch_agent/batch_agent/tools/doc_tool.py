"""
Autonomous Document Ingestion Tool for Research Papers and SRS documents.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from batch_agent.audit.events import AuditEvent, AuditEventType
from batch_agent.documents.analyzer import DocumentAnalyzer
from batch_agent.tools.base import BaseTool, ToolContext, ToolResult


class DocumentIngestTool(BaseTool):
    """Tool allowing autonomous agent to parse research papers and SRS documents."""

    name = "ingest_document"
    description = (
        "Ingest, parse, and semantically analyze research papers (PDF, LaTeX, Markdown) "
        "or Software Requirements Specifications (SRS, PRD, HTML). "
        "Extracts functional requirements, algorithmic steps, mathematical equations, and architecture parameters."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute path to the research paper or specification document.",
            },
            "mode": {
                "type": "string",
                "enum": ["full", "requirements", "algorithms", "summary"],
                "default": "full",
                "description": "Extraction focus mode.",
            },
        },
        "required": ["path"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        rel_or_abs = params.get("path", "")
        if not rel_or_abs:
            return ToolResult(success=False, error="Parameter 'path' is required.")

        abs_path = (
            rel_or_abs
            if os.path.isabs(rel_or_abs)
            else os.path.join(context.cwd, rel_or_abs)
        )

        if not os.path.exists(abs_path):
            return ToolResult(success=False, error=f"Document file not found: {rel_or_abs}")

        mode = params.get("mode", "full")

        try:
            digest = DocumentAnalyzer.analyze_file(abs_path)

            # Log audit event
            if context.audit_logger:
                context.audit_logger.record_event(
                    AuditEvent(
                        event_type=AuditEventType.DOCUMENT_INGESTION,
                        step_number=context.step_number,
                        title=f"Technical Document Ingested: {digest.title}",
                        description=(
                            f"Parsed {digest.doc_type.value} from '{rel_or_abs}'. "
                            f"Identified {len(digest.requirements)} requirements and "
                            f"{len(digest.algorithms)} algorithm specifications."
                        ),
                        metadata={
                            "document_path": rel_or_abs,
                            "doc_type": digest.doc_type.value,
                            "title": digest.title,
                            "requirements_count": len(digest.requirements),
                            "algorithms_count": len(digest.algorithms),
                            "parameters": digest.key_parameters,
                        },
                    )
                )

            # Format output based on requested mode
            if mode == "requirements":
                req_lines = [f"# Requirements Extracted from: {digest.title}"]
                for r in digest.requirements:
                    req_lines.append(f"- [{r.id}] ({r.priority.upper()}) {r.title}: {r.description}")
                output_str = "\n".join(req_lines)

            elif mode == "algorithms":
                algo_lines = [f"# Algorithms Extracted from: {digest.title}"]
                for a in digest.algorithms:
                    algo_lines.append(f"## {a.name}\n{a.description}")
                    if a.steps:
                        algo_lines.append("Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(a.steps)))
                    if a.pseudocode:
                        algo_lines.append(f"Pseudocode:\n```\n{a.pseudocode}\n```")
                output_str = "\n".join(algo_lines)

            elif mode == "summary":
                output_str = f"# Document: {digest.title} ({digest.doc_type.value})\n\n{digest.summary}"

            else:
                output_str = digest.to_prompt_context(max_chars=3500)

            return ToolResult(
                success=True,
                output=output_str,
                metadata=digest.to_dict(),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to ingest document '{rel_or_abs}': {e}",
            )

