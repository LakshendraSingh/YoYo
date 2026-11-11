"""
Unit and integration tests for Technical Document & Research Paper Ingestion
(PDFs, LaTeX, Markdown SRS documents, and Algorithm specifications).
"""

import os
import tempfile
import unittest
import zlib

from batch_agent.cli import main
from batch_agent.config import (
    AuditConfig,
    BatchAgentConfig,
    LLMProviderType,
    ProcessConfig,
    SandboxConfig,
    SandboxProfile,
)
from batch_agent.core.engine import BatchEngine
from batch_agent.documents.analyzer import DocumentAnalyzer
from batch_agent.documents.models import DocumentType
from batch_agent.documents.parser import DocumentParser
from batch_agent.tools.base import ToolContext
from batch_agent.tools.doc_tool import DocumentIngestTool


class TestDocumentIngestion(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.audit_log = os.path.join(self.ws, "doc_audit.txt")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_markdown_srs_parsing_and_analysis(self):
        srs_content = """# Software Requirements Specification (SRS) for Data Pipeline
## 1. Overview
This system provides high-performance telemetry processing for autonomous batch nodes.

## 2. Functional Requirements
- REQ-01: The system shall parse CSV files and validate schema integrity.
- REQ-02: The system shall calculate mean, standard deviation, and anomaly scores.
- REQ-03: The system shall persist computed results to output_results.json.

## 3. Non-Functional Requirements
- NFR-01: Memory consumption must not exceed 512MB during streaming.
- NFR-02: Total execution time shall be under 60 seconds.

## 4. Configuration Parameters
batch_size: 128
timeout: 30
threshold: 0.95
"""
        srs_path = os.path.join(self.ws, "pipeline_srs.md")
        with open(srs_path, "w", encoding="utf-8") as f:
            f.write(srs_content)

        digest = DocumentAnalyzer.analyze_file(srs_path)

        self.assertEqual(digest.doc_type, DocumentType.SRS_SPECIFICATION)
        self.assertIn("Software Requirements Specification", digest.title)
        self.assertGreaterEqual(len(digest.requirements), 3)

        req_ids = [r.id for r in digest.requirements]
        self.assertIn("REQ-01", req_ids)
        self.assertIn("REQ-02", req_ids)
        self.assertIn("NFR-01", req_ids)

        self.assertEqual(digest.key_parameters.get("batch_size"), "128")
        self.assertEqual(digest.key_parameters.get("threshold"), "0.95")

    def test_latex_research_paper_parsing_and_analysis(self):
        latex_content = r"""\documentclass{article}
\begin{document}
\title{Optimized Matrix Factorization for Heterogeneous Batch Clusters}
\maketitle

\begin{abstract}
We present a vectorized decomposition methodology reducing computational complexity in headless nodes.
\end{abstract}

\section{Introduction}
Large-scale batch computing requires robust algorithmic procedures without human supervision \cite{ref1}.

\section{Methodology}
\begin{algorithm}
Algorithm 1: Adaptive Matrix Regularization
Input: Data matrix X, regularization parameter lambda
Output: Low-rank factors U and V
1. Initialize factors U and V randomly
2. Compute gradient del_L = X - U * V^T
3. Update U = U - lr * del_L * V
4. Repeat until convergence
\end{algorithm}

\section{Hyperparameters}
learning_rate = 0.001
epochs = 50
hidden_dim = 64
\end{document}
"""
        latex_path = os.path.join(self.ws, "matrix_paper.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        digest = DocumentAnalyzer.analyze_file(latex_path)

        self.assertEqual(digest.doc_type, DocumentType.RESEARCH_PAPER)
        self.assertIn("Optimized Matrix Factorization", digest.title)
        self.assertIn("vectorized decomposition", digest.summary)
        self.assertGreaterEqual(len(digest.algorithms), 1)

        algo = digest.algorithms[0]
        self.assertIn("Adaptive Matrix Regularization", algo.name)
        self.assertTrue(len(algo.steps) >= 3)
        self.assertEqual(digest.key_parameters.get("learning_rate"), "0.001")
        self.assertEqual(digest.key_parameters.get("hidden_dim"), "64")

    def test_native_pdf_stream_extraction(self):
        """Create a minimal synthetic PDF file with text streams to verify native PDF parser."""
        pdf_text = "(Hello Autonomous Research Paper) Tj\n(Testing Algorithm 1 Methodology) Tj"
        compressed_stream = zlib.compress(pdf_text.encode("latin-1"))

        pdf_binary = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Length " + str(len(compressed_stream)).encode("ascii") + b" /Filter /FlateDecode >>\n"
            b"stream\n" + compressed_stream + b"\nendstream\nendobj\n"
            b"trailer\n<< /Root 1 0 R >>\n%%EOF"
        )

        pdf_path = os.path.join(self.ws, "synthetic_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_binary)

        extracted_text = DocumentParser.parse_file(pdf_path)
        self.assertIn("Hello Autonomous Research Paper", extracted_text)
        self.assertIn("Testing Algorithm 1 Methodology", extracted_text)

    def test_document_ingest_tool_execution(self):
        doc_path = os.path.join(self.ws, "spec.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "# PRD: Automated Telemetry Ingest\n"
                "## Requirements\n"
                "- REQ-101: System shall record execution traces\n"
                "- REQ-102: System must generate JSON report\n"
            )

        tool = DocumentIngestTool()
        context = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=None,  # type: ignore
            tracker=None,  # type: ignore
            runner=None,   # type: ignore
        )

        # Mode: full
        res_full = tool.run({"path": "spec.md", "mode": "full"}, context)
        self.assertTrue(res_full.success)
        self.assertIn("Automated Telemetry Ingest", res_full.output)
        self.assertIn("REQ-101", res_full.output)

        # Mode: requirements
        res_req = tool.run({"path": "spec.md", "mode": "requirements"}, context)
        self.assertTrue(res_req.success)
        self.assertIn("REQ-101", res_req.output)
        self.assertIn("REQ-102", res_req.output)

    def test_end_to_end_batch_execution_with_srs_document(self):
        """Verify YoYo ingests SRS doc from associated files and completes the job successfully."""
        srs_file = os.path.join(self.ws, "spec_srs.md")
        with open(srs_file, "w", encoding="utf-8") as f:
            f.write(
                "# System Specification\n"
                "The agent must process data and verify outputs.\n"
                "- REQ-01: Calculate aggregate statistics\n"
                "- REQ-02: Verify zero regression errors\n"
            )

        # Sample data
        data_file = os.path.join(self.ws, "data.csv")
        with open(data_file, "w", encoding="utf-8") as f:
            f.write("a,b\n1,2\n3,4\n")

        config = BatchAgentConfig(
            job_id="srs-test-job-001",
            goal="Implement data processing conforming to spec_srs.md",
            associated_files=["spec_srs.md", "data.csv"],
            workspace_root=self.ws,
            max_steps=6,
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(enabled=True, profile=SandboxProfile.WORKSPACE, workspace_root=self.ws),
            process=ProcessConfig(default_timeout_seconds=10),
            audit=AuditConfig(log_file_path=self.audit_log, flush_immediately=True),
        )

        engine = BatchEngine(config)
        code = engine.run()

        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.audit_log))

        with open(self.audit_log, "r", encoding="utf-8") as f:
            log_text = f.read()

        # Verify document ingestion was logged in audit trail
        self.assertIn("[DOCUMENT_INGESTION]", log_text)
        self.assertIn("System Specification", log_text)
        self.assertIn("Final Job Status   : SUCCESS", log_text)


if __name__ == "__main__":
    unittest.main()

