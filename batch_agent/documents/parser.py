"""
Robust, zero-dependency document parser for research papers (PDF, LaTeX, Markdown)
and software requirements specifications (SRS, PRDs, Text, HTML).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import zlib
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentParser:
    """Multi-format parser extracting text, structure, and formulas from documents."""

    @classmethod
    def parse_file(cls, file_path: str) -> str:
        """Parse document and return normalized text content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext in (".tex", ".latex", ".bib"):
            return cls.parse_latex(file_path)
        elif ext in (".md", ".markdown", ".rst", ".txt"):
            return cls.parse_text_file(file_path)
        elif ext in (".html", ".htm"):
            return cls.parse_html(file_path)
        elif ext in (".json", ".yaml", ".yml"):
            return cls.parse_text_file(file_path)
        else:
            # Fallback to general text reading
            return cls.parse_text_file(file_path)

    @classmethod
    def parse_text_file(cls, file_path: str) -> str:
        """Read standard text/markdown files with robust encoding fallbacks."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # Binary fallback with replacement
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")

    @classmethod
    def parse_latex(cls, file_path: str) -> str:
        """Parse LaTeX scientific papers and strip typesetting boilerplate."""
        raw = cls.parse_text_file(file_path)

        # Remove LaTeX comments (% ...)
        clean = re.sub(r"(?<!\\)%.*$", "", raw, flags=re.MULTILINE)

        # Convert title and clean document boilerplate
        clean = re.sub(r"\\title\*?\{([^}]+)\}", r"\n\n# \1\n", clean)
        clean = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}", "", clean)
        clean = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}", "", clean)
        clean = re.sub(r"\\(?:author|date|maketitle)\*?(?:\{[^}]*\})?", "", clean)

        # Preserve section headers in clean format: \section{Title} -> ## Title
        clean = re.sub(r"\\section\*?\{([^}]+)\}", r"\n\n## \1\n", clean)
        clean = re.sub(r"\\subsection\*?\{([^}]+)\}", r"\n\n### \1\n", clean)
        clean = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"\n\n#### \1\n", clean)

        # Normalize common environments
        clean = re.sub(r"\\begin\{abstract\}", r"\n## Abstract\n", clean)
        clean = re.sub(r"\\end\{abstract\}", r"\n", clean)

        clean = re.sub(r"\\begin\{algorithm\}", r"\n\n", clean)
        clean = re.sub(r"\\end\{algorithm\}", r"\n\n", clean)

        # Replace citations \cite{...} and references \ref{...} with simplified markers
        clean = re.sub(r"\\cite\{[^}]+\}", "[ref]", clean)
        clean = re.sub(r"\\ref\{[^}]+\}", "[fig/sec]", clean)

        # Remove formatting tags like \textbf{text} -> text, \textit{text} -> text
        clean = re.sub(r"\\(?:textbf|textit|emph|underline|text)\{([^}]+)\}", r"\1", clean)

        # Strip remaining command backslashes while keeping math expressions
        clean = re.sub(r"\\[a-zA-Z]+(?:\*|\b)", " ", clean)

        return clean.strip()

    @classmethod
    def parse_html(cls, file_path: str) -> str:
        """Strip HTML markup and extract readable specification text."""
        raw = cls.parse_text_file(file_path)
        # Remove script and style elements
        clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # Convert headers and paragraph breaks
        clean = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n\n## \1\n", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<li[^>]*>(.*?)</li>", r"\n* \1", clean, flags=re.IGNORECASE)
        # Strip all other HTML tags
        clean = re.sub(r"<[^>]+>", " ", clean)
        return re.sub(r"\n{3,}", "\n\n", clean).strip()

    @classmethod
    def parse_pdf(cls, file_path: str) -> str:
        """
        Multi-tier PDF parser:
        Tier 1: Uses installed python libraries (pypdf, pymupdf, pdfplumber) if present.
        Tier 2: Uses system utility (pdftotext) if available on host.
        Tier 3: Built-in native standard-library PDF object/stream extractor (zlib FlateDecode).
        """
        # Tier 1: Try Python libraries
        text = cls._try_pypdf(file_path)
        if text and len(text.strip()) > 50:
            return text

        # Tier 2: Try pdftotext CLI utility
        text = cls._try_pdftotext(file_path)
        if text and len(text.strip()) > 50:
            return text

        # Tier 3: Native standard-library stream extraction
        return cls._native_pdf_extract(file_path)

    @classmethod
    def _try_pypdf(cls, file_path: str) -> Optional[str]:
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(file_path)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            if pages_text:
                return "\n\n".join(pages_text)
        except Exception:
            pass

        try:
            import fitz  # type: ignore (PyMuPDF)
            doc = fitz.open(file_path)
            pages_text = [page.get_text() for page in doc]
            if pages_text:
                return "\n\n".join(pages_text)
        except Exception:
            pass

        return None

    @classmethod
    def _try_pdftotext(cls, file_path: str) -> Optional[str]:
        pdftotext_bin = shutil.which("pdftotext")
        if not pdftotext_bin:
            return None
        try:
            res = subprocess.run(
                [pdftotext_bin, "-layout", file_path, "-"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if res.returncode == 0 and res.stdout:
                return res.stdout
        except Exception:
            pass
        return None

    @classmethod
    def _native_pdf_extract(cls, file_path: str) -> str:
        """
        Native standard-library PDF text and stream decompressor.
        Extracts uncompressed and FlateDecode compressed stream blocks,
        parsing string tokens (text in parentheses) and TJ text arrays.
        """
        with open(file_path, "rb") as f:
            content = f.read()

        extracted_chunks = []

        # Find all stream ... endstream blocks
        stream_matches = list(re.finditer(b"stream[\r\n]+(.*?)[\r\n]+endstream", content, re.DOTALL))

        for m in stream_matches:
            raw_stream = m.group(1)
            decompressed: Optional[bytes] = None

            # Attempt Flate decompression
            try:
                decompressed = zlib.decompress(raw_stream)
            except Exception:
                try:
                    # zlib with wbits for header bypass
                    decompressed = zlib.decompress(raw_stream, -15)
                except Exception:
                    decompressed = raw_stream

            if decompressed:
                try:
                    text_content = decompressed.decode("latin-1", errors="ignore")
                    extracted_chunks.append(cls._extract_pdf_tokens(text_content))
                except Exception:
                    pass

        # If stream extraction yielded valid text
        full_text = "\n".join([c for c in extracted_chunks if c.strip()])
        if len(full_text.strip()) > 30:
            return full_text

        # Fallback: scan printable ASCII characters directly from file bytes
        printable = re.findall(rb"[A-Za-z0-9 ,;.:!?'\"()\[\]\-_\n\t]{4,}", content)
        fallback_lines = []
        for p in printable:
            try:
                s = p.decode("ascii").strip()
                # Exclude PDF structural keywords
                if not any(k in s for k in ("obj", "endobj", "stream", "endstream", "xref", "trailer", "/Type", "/Font")):
                    fallback_lines.append(s)
            except Exception:
                pass

        return "\n".join(fallback_lines)

    @classmethod
    def _extract_pdf_tokens(cls, stream_text: str) -> str:
        """Extract text tokens from a PDF content stream."""
        lines = []
        # Find Tj operations: (string) Tj
        for match in re.finditer(r"\((.*?)\)\s*Tj", stream_text):
            lines.append(match.group(1))

        # Find TJ array operations: [ (text) 10 (text) ] TJ
        for match in re.finditer(r"\[(.*?)\]\s*TJ", stream_text):
            inner = match.group(1)
            parts = re.findall(r"\((.*?)\)", inner)
            if parts:
                lines.append("".join(parts))

        return " ".join(lines)
