"""
AST and Structural Symbol Inspection Tool.
Extracts functions, classes, methods, docstrings, and imports for Python (via built-in ast)
and C/C++ source/header files (via zero-dependency structural parser).
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from batch_agent.tools.base import BaseTool, ToolContext, ToolResult

CPP_EXTENSIONS = frozenset({".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx"})
_CPP_CONTROL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "else", "return",
    "sizeof", "alignof", "decltype", "static_assert", "typedef",
})


class CodeASTInspectTool(BaseTool):
    """Analyzes Python and C/C++ files to extract classes, structs, functions, and imports."""

    name = "ast_inspect"
    description = (
        "Parse a Python (.py) or C/C++ (.c, .cpp, .cc, .h, .hpp) source file to extract "
        "defined classes/structs, methods, functions, parameters, and imports/includes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the Python or C/C++ file"},
            "query": {"type": "string", "description": "Optional case-insensitive symbol filter (e.g. function or class name)"},
        },
        "required": ["path"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = params.get("path", "")
        query = (params.get("query") or "").strip().lower()
        allowed, reason = context.sandbox.validate_read(path, context.cwd)
        if not allowed:
            return ToolResult(success=False, error=reason)

        abs_path = os.path.abspath(os.path.join(context.cwd, path)) if not os.path.isabs(path) else path
        if not os.path.exists(abs_path):
            return ToolResult(success=False, error=f"File not found: {path}")

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")

        ext = os.path.splitext(abs_path)[1].lower()
        if ext in CPP_EXTENSIONS:
            imports, classes, functions = self._inspect_cpp(source)
        else:
            result = self._inspect_python(source, path)
            if isinstance(result, ToolResult):
                return result
            imports, classes, functions = result

        if query:
            imports = [i for i in imports if query in i.lower()]
            classes = [c for c in classes if query in c.lower()]
            functions = [fn for fn in functions if query in fn.lower()]

        output_parts = [f"=== AST Symbol Summary for {path}" + (f" (filter='{query}')" if query else "") + " ==="]
        if imports:
            output_parts.append("\nImports:\n" + "\n".join(f"  {i}" for i in imports))
        if classes:
            output_parts.append("\nClasses:\n" + "\n".join(f"  {c}" for c in classes))
        if functions:
            output_parts.append("\nTop-level Functions:\n" + "\n".join(f"  {fn}" for fn in functions))

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata={"classes_count": len(classes), "functions_count": len(functions), "imports_count": len(imports)},
        )

    @staticmethod
    def _inspect_python(source: str, path: str) -> Tuple[List[str], List[str], List[str]] | ToolResult:
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Syntax error at line {e.lineno}, col {e.offset}: {e.msg}",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to parse AST: {e}")

        imports: List[str] = []
        classes: List[str] = []
        functions: List[str] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(a.name for a in node.names)
                imports.append(f"from {module} import {names}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                doc = ast.get_docstring(node)
                doc_summary = f" - '{doc.splitlines()[0]}'" if doc else ""
                functions.append(f"def {node.name}({', '.join(args)}): (Line {node.lineno}){doc_summary}")
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                doc_summary = f" - '{doc.splitlines()[0]}'" if doc else ""
                method_names = []
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_args = [a.arg for a in sub.args.args]
                        method_names.append(f"    def {sub.name}({', '.join(method_args)}): (Line {sub.lineno})")
                classes.append(f"class {node.name}: (Line {node.lineno}){doc_summary}\n" + "\n".join(method_names))

        return imports, classes, functions

    @staticmethod
    def _strip_cpp_comments(source: str) -> str:
        """Strip C/C++ block and line comments while preserving newline offsets."""
        pattern = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

        def _replacer(match: re.Match) -> str:
            text = match.group(0)
            if text.startswith("//") or text.startswith("/*"):
                return "\n" * text.count("\n")
            return text

        return pattern.sub(_replacer, source)

    @classmethod
    def _inspect_cpp(cls, source: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract #includes, classes/structs, methods, and functions from C/C++ source."""
        imports: List[str] = []
        class_entries: List[Tuple[str, int, List[str]]] = []
        functions: List[str] = []

        # Collect #includes from raw lines
        for line in source.splitlines():
            m_inc = re.match(r"^\s*(#\s*include\s+[<\"][^>\"]+[>\"])", line)
            if m_inc:
                imports.append(m_inc.group(1))

        cleaned = cls._strip_cpp_comments(source)
        lines = cleaned.splitlines()

        class_re = re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?(class|struct)\s+([A-Za-z_]\w*)\b([^;]*)$")
        func_re = re.compile(
            r"^\s*(?:template\s*<[^>]*>\s*)?"
            r"(?:(?:inline|static|virtual|explicit|constexpr|extern|friend)\s+)*"
            r"(?:([\w:*&<>\s]+?)\s+)?([~A-Za-z_]\w*(?:::[~A-Za-z_]\w*)*)\s*\(([^;{}]*)\)"
            r"(?:\s*(?:const|override|final|noexcept))*\s*(\{|;)?\s*$"
        )

        brace_depth = 0
        # Stack of (class_index, entry_brace_depth)
        active_classes: List[Tuple[int, int]] = []
        pending_class_idx: Optional[int] = None

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Detect class/struct definition (skip forward declarations ending with ';')
            m_cls = class_re.match(line)
            if m_cls and not stripped.endswith(";"):
                kind, cname = m_cls.group(1), m_cls.group(2)
                idx = len(class_entries)
                class_entries.append((f"{kind} {cname}", lineno, []))
                if "{" in line:
                    active_classes.append((idx, brace_depth + line.count("{") - line.count("}")))
                    brace_depth += line.count("{") - line.count("}")
                    continue
                pending_class_idx = idx

            # Detect function/method declaration or definition
            elif not stripped.startswith(("return ", "using ", "typedef ")):
                m_fn = func_re.match(line)
                if m_fn:
                    ret_type = (m_fn.group(1) or "").strip()
                    fname = m_fn.group(2).strip()
                    fparams = " ".join(m_fn.group(3).split())
                    trailing = m_fn.group(4)
                    base_name = fname.split("::")[-1].lstrip("~")
                    # Require either a return type, destructor/constructor in class, or '{'/';'
                    if base_name not in _CPP_CONTROL_KEYWORDS and (ret_type or active_classes or trailing in ("{", ";")):
                        sig = f"{ret_type + ' ' if ret_type else ''}{fname}({fparams}) (Line {lineno})"
                        if active_classes and "::" not in fname:
                            cls_idx, _ = active_classes[-1]
                            class_entries[cls_idx][2].append(f"    {sig}")
                        else:
                            functions.append(sig)

            brace_depth += line.count("{") - line.count("}")

            if pending_class_idx is not None and "{" in line:
                active_classes.append((pending_class_idx, brace_depth))
                pending_class_idx = None

            while active_classes and brace_depth < active_classes[-1][1]:
                active_classes.pop()

        classes = []
        for header, lineno, methods in class_entries:
            block = f"{header}: (Line {lineno})"
            if methods:
                block += "\n" + "\n".join(methods)
            classes.append(block)

        return imports, classes, functions
