"""
Data models for ingested research papers, SRS documents, and technical specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentType(str, Enum):
    RESEARCH_PAPER = "RESEARCH_PAPER"
    SRS_SPECIFICATION = "SRS_SPECIFICATION"
    ALGORITHM_GUIDE = "ALGORITHM_GUIDE"
    TECHNICAL_SPEC = "TECHNICAL_SPEC"
    GENERAL_DOC = "GENERAL_DOC"


@dataclass
class RequirementSpec:
    """Individual specification or requirement extracted from an SRS document."""
    id: str
    title: str
    description: str
    category: str = "functional"  # functional, non-functional, interface, constraint
    priority: str = "high"        # high, medium, low

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
        }


@dataclass
class AlgorithmSpec:
    """Algorithm, formula, or methodology extracted from a research paper or technical spec."""
    name: str
    description: str
    steps: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    pseudocode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "equations": self.equations,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "pseudocode": self.pseudocode,
        }


@dataclass
class DocumentDigest:
    """
    Structured summary and semantic representation of an ingested document
    (Research Paper, SRS document, Architecture RFC, or Algorithm Specification).
    """
    file_path: str
    doc_type: DocumentType
    title: str
    summary: str
    sections: Dict[str, str] = field(default_factory=dict)
    requirements: List[RequirementSpec] = field(default_factory=list)
    algorithms: List[AlgorithmSpec] = field(default_factory=list)
    key_parameters: Dict[str, str] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "doc_type": self.doc_type.value,
            "title": self.title,
            "summary": self.summary,
            "sections": list(self.sections.keys()),
            "requirements": [r.to_dict() for r in self.requirements],
            "algorithms": [a.to_dict() for a in self.algorithms],
            "key_parameters": self.key_parameters,
            "total_chars": len(self.raw_text),
        }

    def to_prompt_context(self, max_chars: int = 4000) -> str:
        """Format a clear, high-density context block for LLM prompts and planning."""
        lines = [
            f"=== INGESTED DOCUMENT: {self.title} ({self.doc_type.value}) ===",
            f"Source File: {self.file_path}",
            f"Overview Summary:\n{self.summary[:500]}",
        ]

        if self.requirements:
            lines.append("\nExtracted Requirements & Constraints:")
            for r in self.requirements[:10]:
                lines.append(f"  * [{r.id}] {r.title}: {r.description[:120]}")

        if self.algorithms:
            lines.append("\nExtracted Algorithms & Methodology:")
            for a in self.algorithms[:5]:
                lines.append(f"  * Algorithm: {a.name}")
                if a.inputs:
                    lines.append(f"    - Inputs : {', '.join(a.inputs)}")
                if a.outputs:
                    lines.append(f"    - Outputs: {', '.join(a.outputs)}")
                if a.steps:
                    lines.append("    - Steps  :")
                    for step in a.steps[:5]:
                        lines.append(f"      1. {step}")
                if a.pseudocode:
                    lines.append(f"    - Pseudocode:\n{a.pseudocode[:300]}")

        if self.key_parameters:
            lines.append("\nKey Hyperparameters & Architecture Configs:")
            for k, v in list(self.key_parameters.items())[:10]:
                lines.append(f"  * {k}: {v}")

        res = "\n".join(lines)
        if len(res) > max_chars:
            return res[:max_chars] + "\n... [Document context truncated]"
        return res

