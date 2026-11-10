"""
AST Symbol and Structural Inspection Tool.
Extracts functions, classes, methods, docstrings, and imports using Python's built-in ast module.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional

from batch_agent.tools.base import BaseTool, ToolContext, ToolResult


class CodeASTInspectTool(BaseTool):
    """Analyzes Python files to extract classes, functions, imports, and docstrings."""

    name = "ast_inspect"
    description = (
        "Parse a Python source file into an AST to extract defined classes, methods, "
        "functions, parameters, docstrings, and imported modules."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the Python file"},
        },
        "required": ["path"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = params.get("path", "")
        allowed, reason = context.sandbox.validate_read(path, context.cwd)
        if not allowed:
            return ToolResult(success=False, error=reason)

        abs_path = os.path.abspath(os.path.join(context.cwd, path)) if not os.path.isabs(path) else path
        if not os.path.exists(abs_path):
            return ToolResult(success=False, error=f"File not found: {path}")

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()

            tree = ast.parse(source, filename=path)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Syntax error at line {e.lineno}, col {e.offset}: {e.msg}",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to parse AST: {e}")

        # Extract symbols
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

        output_parts = [f"=== AST Symbol Summary for {path} ==="]
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

