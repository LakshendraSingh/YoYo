"""
Semantic document analyzer for extracting requirements, algorithms, and parameters
from research papers, software requirements specifications (SRS), and technical PRDs.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

from batch_agent.documents.models import (
    AlgorithmSpec,
    DocumentDigest,
    DocumentType,
    RequirementSpec,
)
from batch_agent.documents.parser import DocumentParser


class DocumentAnalyzer:
    """Analyzes technical papers and SRS documents to extract actionable specifications."""

    @classmethod
    def analyze_file(cls, file_path: str) -> DocumentDigest:
        """Parse file and return structured DocumentDigest."""
        text = DocumentParser.parse_file(file_path)
        return cls.analyze_text(file_path, text)

    @classmethod
    def analyze_text(cls, file_path: str, text: str) -> DocumentDigest:
        """Extract structured entities, requirements, and algorithms from text."""
        doc_type = cls.classify_document(file_path, text)
        title = cls.extract_title(file_path, text)
        sections = cls.extract_sections(text)
        summary = cls.extract_summary(sections, text)
        requirements = cls.extract_requirements(text)
        algorithms = cls.extract_algorithms(text)
        params = cls.extract_parameters(text)

        return DocumentDigest(
            file_path=file_path,
            doc_type=doc_type,
            title=title,
            summary=summary,
            sections=sections,
            requirements=requirements,
            algorithms=algorithms,
            key_parameters=params,
            raw_text=text,
        )

    @classmethod
    def classify_document(cls, file_path: str, text: str) -> DocumentType:
        """Classify document into Research Paper, SRS, Algorithm Guide, or Technical Spec."""
        lower_path = file_path.lower()
        lower_text = text[:4000].lower()

        # Check path hints
        if "srs" in lower_path or "requirements" in lower_path or "spec" in lower_path:
            return DocumentType.SRS_SPECIFICATION
        if "paper" in lower_path or "arxiv" in lower_path or "preprint" in lower_path:
            return DocumentType.RESEARCH_PAPER

        # Score text features
        srs_signals = sum(
            1 for w in (
                "software requirements specification",
                "functional requirement",
                "non-functional requirement",
                "system shall",
                "acceptance criteria",
                "use case",
                "user story",
                "product requirements document",
                "scope of work",
            ) if w in lower_text
        )

        paper_signals = sum(
            1 for w in (
                "abstract",
                "methodology",
                "experimental setup",
                "ablation study",
                "prior work",
                "related work",
                "state-of-the-art",
                "arxiv",
                "conference",
                "proceedings",
                "theorem",
                "equation",
            ) if w in lower_text
        )

        algo_signals = sum(
            1 for w in (
                "algorithm 1",
                "pseudocode",
                "input:",
                "output:",
                "procedure",
                "time complexity",
            ) if w in lower_text
        )

        if srs_signals >= 2 or ("functional" in lower_text and "requirement" in lower_text):
            return DocumentType.SRS_SPECIFICATION
        if paper_signals >= 2 or ("abstract" in lower_text and "method" in lower_text):
            return DocumentType.RESEARCH_PAPER
        if algo_signals >= 2:
            return DocumentType.ALGORITHM_GUIDE

        return DocumentType.TECHNICAL_SPEC

    @classmethod
    def extract_title(cls, file_path: str, text: str) -> str:
        """Extract clean document title."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Check for markdown # Header
        for line in lines[:10]:
            if line.startswith("# "):
                return line.lstrip("# ").strip()

        # Check for Title: Header
        for line in lines[:10]:
            if line.lower().startswith("title:"):
                return line.split(":", 1)[1].strip()

        # First prominent non-keyword line
        for line in lines[:5]:
            if len(line) > 5 and not any(k in line.lower() for k in ("abstract", "ieee", "arxiv", "page")):
                return line[:100].strip()

        return os.path.basename(file_path)

    @classmethod
    def extract_sections(cls, text: str) -> Dict[str, str]:
        """Split text into named sections based on Markdown/Text headers."""
        sections: Dict[str, str] = {}
        current_section = "Introduction"
        current_lines: List[str] = []

        for line in text.splitlines():
            header_match = re.match(r"^(?:#{1,4}\s+|\[SECTION\s+\d+:?\s*\]|\d+\.\s+)([A-Za-z0-9\s\-_/]+)$", line.strip())
            if header_match:
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                    current_lines = []
                current_section = header_match.group(1).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        return sections

    @classmethod
    def extract_summary(cls, sections: Dict[str, str], text: str) -> str:
        """Locate executive summary or abstract."""
        for name, content in sections.items():
            if any(k in name.lower() for k in ("abstract", "overview", "executive summary", "summary", "scope")):
                if len(content) > 30:
                    return content[:1000].strip()

        # Fallback to initial paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        if paragraphs:
            return paragraphs[0][:800]
        return text[:400].strip()

    @classmethod
    def extract_requirements(cls, text: str) -> List[RequirementSpec]:
        """Extract functional/non-functional requirements from text."""
        requirements: List[RequirementSpec] = []
        seen_ids = set()

        # Pattern 1: REQ-01 / FR-01 / SR-1: Title - Description
        req_pattern = re.compile(
            r"(?:^|\n)\s*(?:[-*]\s*)?((?:REQ|FR|NFR|SR|SPEC)[-_]?\d+)\s*[:.-]?\s*([^\n\r]+)",
            re.IGNORECASE,
        )
        for m in req_pattern.finditer(text):
            req_id = m.group(1).upper()
            if req_id in seen_ids:
                continue
            seen_ids.add(req_id)
            full_line = m.group(2).strip()
            if " - " in full_line:
                title, desc = full_line.split(" - ", 1)
            elif ": " in full_line:
                title, desc = full_line.split(": ", 1)
            else:
                title, desc = full_line, full_line
            category = "non-functional" if "nfr" in req_id.lower() else "functional"
            requirements.append(
                RequirementSpec(
                    id=req_id,
                    title=title.strip(),
                    description=desc.strip(),
                    category=category,
                    priority="high",
                )
            )

        # Pattern 2: "The system shall ..." or "The agent must ..."
        shall_pattern = re.compile(
            r"(?:^|\n)\s*(?:[-*]\s*)?(?:The\s+)?(system|agent|pipeline|model|algorithm)\s+(shall|must|should)\s+([^\n\.\;]+[\.\;]?)",
            re.IGNORECASE,
        )
        for idx, m in enumerate(shall_pattern.finditer(text), start=len(requirements) + 1):
            subject = m.group(1).capitalize()
            verb = m.group(2).lower()
            predicate = m.group(3).strip()
            req_id = f"REQ-AUTO-{idx}"
            if req_id in seen_ids:
                continue
            seen_ids.add(req_id)
            priority = "high" if verb == "must" or verb == "shall" else "medium"
            requirements.append(
                RequirementSpec(
                    id=req_id,
                    title=f"{subject} {verb} {predicate[:60]}",
                    description=f"{subject} {verb} {predicate}",
                    category="functional",
                    priority=priority,
                )
            )

        return requirements

    @classmethod
    def extract_algorithms(cls, text: str) -> List[AlgorithmSpec]:
        """Extract algorithms, mathematical procedures, or pseudocode from text."""
        algorithms: List[AlgorithmSpec] = []

        # Find "Algorithm X: Name" blocks
        algo_blocks = re.finditer(
            r"(?:Algorithm\s+(\d+|[A-Za-z]+)\s*[:.-]?\s*([^\n]+))(.*?)(?=(?:Algorithm\s+\d+|##|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        for m in algo_blocks:
            algo_num = m.group(1).strip()
            algo_name = m.group(2).strip() or f"Algorithm {algo_num}"
            body = m.group(3).strip()

            inputs = []
            outputs = []
            steps = []
            equations = []

            for line in body.splitlines():
                line_str = line.strip()
                if not line_str:
                    continue

                if re.match(r"^input\s*:", line_str, re.IGNORECASE):
                    inputs.append(re.sub(r"^input\s*:\s*", "", line_str, flags=re.IGNORECASE))
                elif re.match(r"^output\s*:", line_str, re.IGNORECASE):
                    outputs.append(re.sub(r"^output\s*:\s*", "", line_str, flags=re.IGNORECASE))
                elif re.match(r"^\d+\.\s+", line_str) or re.match(r"^step\s+\d+", line_str, re.IGNORECASE):
                    steps.append(line_str)
                elif any(sym in line_str for sym in ("=", "∑", "∈", "argmin", "argmax", "sqrt", "∇", "exp")):
                    equations.append(line_str)
                else:
                    if len(steps) < 10 and len(line_str) > 10:
                        steps.append(line_str)

            algorithms.append(
                AlgorithmSpec(
                    name=algo_name,
                    description=f"Specification for {algo_name}",
                    steps=steps,
                    equations=equations,
                    inputs=inputs,
                    outputs=outputs,
                    pseudocode=body[:600],
                )
            )

        # Fallback: if no formal "Algorithm X" blocks found, check for Methodology or Formula blocks
        if not algorithms:
            method_match = re.search(r"##\s*(?:Methodology|Algorithm|Proposed Method|Mathematical Formulation)(.*?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE)
            if method_match:
                m_body = method_match.group(1).strip()
                steps = [line.strip() for line in m_body.splitlines() if line.strip() and len(line.strip()) > 15][:5]
                algorithms.append(
                    AlgorithmSpec(
                        name="Proposed Method",
                        description="Methodology extracted from document section",
                        steps=steps,
                        pseudocode=m_body[:500],
                    )
                )

        return algorithms

    @classmethod
    def extract_parameters(cls, text: str) -> Dict[str, str]:
        """Extract hyperparameters, configuration options, and constants."""
        params: Dict[str, str] = {}

        # Key: Value pattern or key = value
        param_patterns = [
            r"(?:^|\n)\s*([A-Za-z0-9_]{3,25})\s*[:=]\s*([0-9.eE\-]+|[A-Za-z0-9_/\.\-]+)",
        ]

        target_keywords = (
            "batch_size", "learning_rate", "lr", "epochs", "iterations",
            "hidden_dim", "embedding_size", "dropout", "weight_decay",
            "threshold", "tolerance", "timeout", "workers", "gpu", "port",
        )

        for pat in param_patterns:
            for m in re.finditer(pat, text):
                k = m.group(1).strip()
                v = m.group(2).strip()
                if any(tk in k.lower() for tk in target_keywords) or len(params) < 15:
                    if k.lower() not in params:
                        params[k] = v

        return params
