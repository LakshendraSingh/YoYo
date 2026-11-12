"""
Filesystem tools for the autonomous batch agent.
Includes resilient multi-tier fuzzy patching to eliminate fragile code editing failures.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from batch_agent.tools.base import BaseTool, ToolContext, ToolResult

DEFAULT_IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    "venv", ".venv", "env", ".env", "dist", "build", "target",
    "cmake-build-debug", "cmake-build-release",
    "vendor", "coverage", ".tox", ".nox", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", ".yoyo_state",
})


def _load_gitignore_patterns(workspace_root: str) -> List[str]:
    """Load ignore patterns from .gitignore at workspace_root if present."""
    gitignore_path = os.path.join(workspace_root, ".gitignore")
    patterns: List[str] = []
    if not os.path.isfile(gitignore_path):
        return patterns
    try:
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                patterns.append(line.rstrip("/"))
    except Exception:
        pass
    return patterns


def _is_ignored_entry(name: str, rel_path: str, is_dir: bool, gitignore_patterns: List[str]) -> bool:
    """Return True if a file or directory should be excluded from workspace traversal."""
    if name.startswith("."):
        return True
    if is_dir and (name in DEFAULT_IGNORED_DIRS or name.endswith(".egg-info")):
        return True
    rel_norm = rel_path.replace(os.sep, "/")
    for pat in gitignore_patterns:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_norm, pat):
            return True
    return False


def _walk_workspace_files(start_dir: str, workspace_root: str) -> Iterator[Tuple[str, str, str]]:
    """Yield (full_path, rel_to_start, filename) while pruning build/venv/gitignored directories."""
    patterns = _load_gitignore_patterns(workspace_root)
    for root, dirs, files in os.walk(start_dir):
        rel_ws = os.path.relpath(root, workspace_root)
        dirs[:] = sorted(
            d for d in dirs
            if not _is_ignored_entry(
                d,
                d if rel_ws == "." else os.path.join(rel_ws, d),
                is_dir=True,
                gitignore_patterns=patterns,
            )
        )
        rel_start = os.path.relpath(root, start_dir)
        for f in sorted(files):
            rel_file_ws = f if rel_ws == "." else os.path.join(rel_ws, f)
            if _is_ignored_entry(f, rel_file_ws, is_dir=False, gitignore_patterns=patterns):
                continue
            full_path = os.path.join(root, f)
            rel_file_start = f if rel_start == "." else os.path.join(rel_start, f)
            yield full_path, rel_file_start, f


def _find_path_candidates(path: str, context: ToolContext) -> List[str]:
    """Search non-ignored workspace files for basename/stem matches to recover from shorthand paths."""
    basename = os.path.basename(path)
    stem = os.path.splitext(basename)[0]
    ext = os.path.splitext(basename)[1]
    candidates: List[str] = []
    for full_path, _, f in _walk_workspace_files(context.workspace_root, context.workspace_root):
        f_stem = os.path.splitext(f)[0]
        if f == basename or stem == f_stem or (stem in f_stem and (not ext or f.endswith(ext))):
            candidates.append(os.path.relpath(full_path, context.cwd))
    return candidates


class FileReadTool(BaseTool):
    """Reads content of a file within sandbox boundaries."""

    name = "file_read"
    description = "Read the contents of a file within the workspace boundary."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path to the file"},
            "start_line": {"type": "integer", "description": "Optional 1-based start line"},
            "end_line": {"type": "integer", "description": "Optional 1-based end line"},
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
            candidates = _find_path_candidates(path, context)
            if len(candidates) > 1 and "test" not in os.path.basename(path).lower():
                non_tests = [c for c in candidates if "test" not in os.path.basename(c).lower()]
                if len(non_tests) == 1:
                    candidates = non_tests
            if len(candidates) == 1:
                path = candidates[0]
                abs_path = os.path.abspath(os.path.join(context.cwd, path))
            elif candidates:
                return ToolResult(
                    success=False,
                    error=f"File '{path}' not found at specified path. Did you mean: {candidates[:3]}?",
                )
            else:
                return ToolResult(success=False, error=f"File not found: {path}")

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            start_line = params.get("start_line", 1)
            end_line = params.get("end_line", len(lines))

            # 1-indexed to slice
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            selected_lines = lines[start_idx:end_idx]

            formatted = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted.append(f"{i:4d} | {line}")

            return ToolResult(
                success=True,
                output="".join(formatted),
                metadata={"total_lines": len(lines), "path": abs_path},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Error reading file: {e}")


class FileWriteTool(BaseTool):
    """Writes full content to a file, creating directories as needed."""

    name = "file_write"
    description = "Write full content to a file within sandbox boundaries."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path to the file"},
            "content": {"type": "string", "description": "Full file content to write"},
        },
        "required": ["path", "content"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = params.get("path", "")
        content = params.get("content", "")

        allowed, reason = context.sandbox.validate_write(path, context.cwd)
        if not allowed:
            return ToolResult(success=False, error=reason)

        abs_path = os.path.abspath(os.path.join(context.cwd, path)) if not os.path.isabs(path) else path
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)

            lines_count = len(content.splitlines())
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes ({lines_count} lines) to '{path}'.",
                metadata={"path": abs_path, "bytes": len(content), "lines": lines_count},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}")


class FileEditTool(BaseTool):
    """Applies surgical edits to a file using resilient multi-tier patching."""

    name = "file_edit"
    description = "Replace a target block of text with replacement content in an existing file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file"},
            "target": {"type": "string", "description": "Existing text block to replace"},
            "replacement": {"type": "string", "description": "New replacement text"},
        },
        "required": ["path", "target", "replacement"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = params.get("path", "")
        target = params.get("target", "")
        replacement = params.get("replacement", "")

        allowed, reason = context.sandbox.validate_write(path, context.cwd)
        if not allowed:
            return ToolResult(success=False, error=reason)

        abs_path = os.path.abspath(os.path.join(context.cwd, path)) if not os.path.isabs(path) else path
        if not os.path.exists(abs_path):
            # Attempt autonomous workspace search for shifted or basename-only path
            candidates = _find_path_candidates(path, context)
            if len(candidates) > 1 and target:
                containing_target = []
                for c in candidates:
                    cand_abs = os.path.abspath(os.path.join(context.cwd, c))
                    try:
                        with open(cand_abs, "r", encoding="utf-8") as tf:
                            if target.strip() in tf.read():
                                containing_target.append(c)
                    except Exception:
                        pass
                if len(containing_target) == 1:
                    candidates = containing_target
                elif not containing_target and "test" not in os.path.basename(path).lower():
                    non_tests = [c for c in candidates if "test" not in os.path.basename(c).lower()]
                    if len(non_tests) == 1:
                        candidates = non_tests

            if len(candidates) == 1:
                path = candidates[0]
                abs_path = os.path.abspath(os.path.join(context.cwd, path))
            elif candidates:
                return ToolResult(
                    success=False,
                    error=f"Target file '{path}' not found at specified path. Did you mean: {candidates[:3]}?",
                )
            else:
                return ToolResult(success=False, error=f"Target file does not exist: {path}")

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                original_text = f.read()

            patched_text, method = self._apply_resilient_patch(original_text, target, replacement)
            if patched_text is None:
                return ToolResult(
                    success=False,
                    error=f"Could not locate target block in '{path}'. Resilient fuzzy matching exhausted.",
                )

            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(patched_text)

            diff = list(difflib.unified_diff(
                original_text.splitlines(keepends=True),
                patched_text.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=2,
            ))

            return ToolResult(
                success=True,
                output=f"Successfully patched '{path}' using {method}.\n" + "".join(diff[:50]),
                metadata={"patch_method": method, "diff_lines": len(diff)},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Error editing file: {e}")

    def _apply_resilient_patch(
        self, original: str, target: str, replacement: str
    ) -> Tuple[Optional[str], str]:
        # Tier 1: Exact string replacement
        if target in original:
            return original.replace(target, replacement, 1), "exact_match"

        # Tier 2: Line-ending normalization (\r\n -> \n)
        norm_orig = original.replace("\r\n", "\n")
        norm_target = target.replace("\r\n", "\n")
        norm_repl = replacement.replace("\r\n", "\n")
        if norm_target in norm_orig:
            return norm_orig.replace(norm_target, norm_repl, 1), "line_ending_normalized"

        # Tier 3: Whitespace trimmed line-by-line matching
        orig_lines = norm_orig.splitlines(keepends=True)
        target_lines = norm_target.splitlines(keepends=True)

        stripped_target = [l.strip() for l in target_lines if l.strip()]
        if not stripped_target:
            return None, "empty_target"

        for i in range(len(orig_lines) - len(target_lines) + 1):
            window = [orig_lines[i + j].strip() for j in range(len(target_lines)) if orig_lines[i + j].strip()]
            if window == stripped_target:
                # Replace the window of lines
                new_lines = orig_lines[:i] + [norm_repl if norm_repl.endswith("\n") else norm_repl + "\n"] + orig_lines[i + len(target_lines):]
                return "".join(new_lines), "whitespace_flexible_match"

        # Tier 4: SequenceMatcher anchor fuzzy search
        target_flat = norm_target.strip()
        best_ratio = 0.0
        best_span: Optional[Tuple[int, int]] = None

        target_len = len(norm_target)
        # Search windows within length variation
        min_win = max(10, int(target_len * 0.8))
        max_win = int(target_len * 1.2) + 1

        for w_len in (target_len, min_win, max_win):
            for i in range(0, len(norm_orig) - w_len + 1, max(1, target_len // 20)):
                chunk = norm_orig[i:i + w_len]
                ratio = difflib.SequenceMatcher(None, target_flat, chunk.strip()).ratio()
                if ratio > best_ratio and ratio >= 0.85:
                    best_ratio = ratio
                    best_span = (i, i + w_len)

        if best_span and best_ratio >= 0.85:
            start, end = best_span
            patched = norm_orig[:start] + norm_repl + norm_orig[end:]
            return patched, f"fuzzy_anchor_match (similarity: {best_ratio:.2f})"

        return None, "exhausted"


class FileListTool(BaseTool):
    """Lists files and directories under workspace root."""

    name = "file_list"
    description = "List files and directories in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory path (default: current)"},
            "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
        },
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        rel_path = params.get("path", ".")
        recursive = params.get("recursive", False)

        allowed, reason = context.sandbox.validate_read(rel_path, context.cwd)
        if not allowed:
            return ToolResult(success=False, error=reason)

        abs_path = os.path.abspath(os.path.join(context.cwd, rel_path)) if not os.path.isabs(rel_path) else rel_path
        if not os.path.exists(abs_path):
            return ToolResult(success=False, error=f"Directory does not exist: {rel_path}")

        entries: List[str] = []
        try:
            if recursive:
                for _, rel_file, _ in _walk_workspace_files(abs_path, context.workspace_root):
                    entries.append(rel_file)
            else:
                patterns = _load_gitignore_patterns(context.workspace_root)
                rel_ws = os.path.relpath(abs_path, context.workspace_root)
                for entry in sorted(os.listdir(abs_path)):
                    full_entry = os.path.join(abs_path, entry)
                    is_dir = os.path.isdir(full_entry)
                    rel_entry = entry if rel_ws == "." else os.path.join(rel_ws, entry)
                    if not _is_ignored_entry(entry, rel_entry, is_dir=is_dir, gitignore_patterns=patterns):
                        entries.append(entry + ("/" if is_dir else ""))

            total_count = len(entries)
            if total_count > 100:
                output_str = "\n".join(entries[:100]) + f"\n... [Truncated: {total_count - 100} additional files omitted. Use 'file_search' to pinpoint specific symbols or modules]"
            else:
                output_str = "\n".join(entries) if entries else "[Directory is empty]"

            return ToolResult(
                success=True,
                output=output_str,
                metadata={"count": total_count, "path": abs_path, "truncated": total_count > 100},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to list directory: {e}")


class FileSearchTool(BaseTool):
    """Searches workspace files for regex patterns."""

    name = "file_search"
    description = "Search for regex or text matches across workspace files."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text pattern or regex to search for"},
            "extension": {"type": "string", "description": "Optional file extension filter (e.g. '.py')"},
        },
        "required": ["pattern"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = params.get("pattern", "")
        ext_filter = params.get("extension", "")

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        matches: List[str] = []
        ws_root = context.workspace_root

        for full_path, rel_path, filename in _walk_workspace_files(ws_root, ws_root):
            if ext_filter and not filename.endswith(ext_filter):
                continue

            # Skip sensitive files
            if context.sandbox.boundary.is_path_sensitive(full_path):
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")
                            if len(matches) >= 100:
                                break
            except Exception:
                pass

            if len(matches) >= 100:
                break

        return ToolResult(
            success=True,
            output="\n".join(matches) if matches else f"No matches found for '{pattern}'.",
            metadata={"match_count": len(matches)},
        )

