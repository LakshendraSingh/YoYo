"""
Environment and Dependency Exploration Tool.
Autonomously surveys sandbox capabilities, detects available system binaries,
identifies installed Python packages, parses codebase import requirements,
and flags missing dependencies for offline batch execution.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from batch_agent.tools.base import BaseTool, ToolContext, ToolResult


@dataclass
class EnvironmentInventory:
    """Complete inventory of sandbox host environment and dependencies."""
    python_version: str
    python_path: str
    installed_packages: Dict[str, str] = field(default_factory=dict)
    system_binaries: Dict[str, str] = field(default_factory=dict)
    gpu_available: bool = False
    gpu_info: str = ""
    required_packages: List[str] = field(default_factory=list)
    missing_packages: List[str] = field(default_factory=list)
    available_packages: List[str] = field(default_factory=list)
    local_wheels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_path": self.python_path,
            "installed_packages_count": len(self.installed_packages),
            "system_binaries": list(self.system_binaries.keys()),
            "gpu_available": self.gpu_available,
            "gpu_info": self.gpu_info,
            "required_packages": self.required_packages,
            "available_packages": self.available_packages,
            "missing_packages": self.missing_packages,
            "local_wheels": self.local_wheels,
        }


class EnvironmentProbe:
    """Scans host environment and workspace dependencies."""

    COMMON_SYSTEM_TOOLS = [
        "git", "gcc", "g++", "make", "pytest", "python3", "pip", "cargo",
        "go", "rustc", "curl", "tar", "gzip", "unzip", "patch", "diff"
    ]

    @classmethod
    def get_stdlib_modules(cls) -> Set[str]:
        if hasattr(sys, "stdlib_module_names"):
            return set(sys.stdlib_module_names)
        # Fallback for Python versions
        return {
            "os", "sys", "time", "math", "re", "json", "urllib", "subprocess",
            "shutil", "tempfile", "pathlib", "logging", "typing", "collections",
            "itertools", "functools", "unittest", "csv", "sqlite3", "hashlib",
            "random", "datetime", "uuid", "socket", "http", "io", "copy",
            "abc", "dataclasses", "enum", "argparse", "difflib", "ast", "signal",
            "atexit", "threading", "multiprocessing", "queue", "traceback"
        }

    @classmethod
    def get_installed_packages(cls) -> Dict[str, str]:
        """Detect all packages installed in the current Python environment."""
        packages = {}
        try:
            for dist in importlib.metadata.distributions():
                name = dist.metadata["Name"].lower() if dist.metadata and dist.metadata["Name"] else dist.name.lower()
                version = dist.version
                packages[name] = version
        except Exception:
            pass
        return packages

    @classmethod
    def get_system_binaries(cls) -> Dict[str, str]:
        """Detect available system utilities on $PATH."""
        binaries = {}
        for tool in cls.COMMON_SYSTEM_TOOLS:
            path = shutil.which(tool)
            if path:
                binaries[tool] = path
        return binaries

    @classmethod
    def check_gpu(cls) -> tuple[bool, str]:
        """Detect GPU/CUDA availability."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA Generic"
                return True, f"CUDA ({name}, {torch.cuda.device_count()} devices)"
        except Exception:
            pass

        # Check nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
                if res.returncode == 0 and res.stdout.strip():
                    return True, f"NVIDIA GPU: {res.stdout.strip()}"
            except Exception:
                pass

        return False, "CPU Only (No GPU detected)"

    @classmethod
    def find_local_wheels(cls, workspace_root: str) -> List[str]:
        """Discover pre-downloaded .whl files in workspace for offline installation."""
        wheels = []
        for root, _, files in os.walk(workspace_root):
            for f in files:
                if f.endswith(".whl"):
                    wheels.append(os.path.relpath(os.path.join(root, f), workspace_root))
        return sorted(wheels)

    @classmethod
    def extract_workspace_imports(cls, workspace_root: str, target_files: Optional[List[str]] = None) -> Set[str]:
        """Extract top-level third-party imported package names from workspace code."""
        stdlib = cls.get_stdlib_modules()
        imported = set()

        files_to_scan = []
        if target_files:
            for f in target_files:
                abs_f = os.path.join(workspace_root, f) if not os.path.isabs(f) else f
                if os.path.isfile(abs_f) and abs_f.endswith(".py"):
                    files_to_scan.append(abs_f)

        if not files_to_scan:
            for root, dirs, files in os.walk(workspace_root):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
                for f in files:
                    if f.endswith(".py"):
                        files_to_scan.append(os.path.join(root, f))

        # Discover local module names to exclude from third-party requirements
        local_modules = {os.path.splitext(os.path.basename(f))[0] for f in files_to_scan}

        for filepath in files_to_scan:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                tree = ast.parse(content, filename=filepath)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            if top and top not in stdlib and top not in local_modules:
                                imported.add(top.lower())
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            top = node.module.split(".")[0]
                            if top and top not in stdlib and top not in local_modules:
                                imported.add(top.lower())
            except Exception:
                pass

        # Check requirements.txt if present
        req_file = os.path.join(workspace_root, "requirements.txt")
        if os.path.isfile(req_file):
            try:
                with open(req_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            pkg = re.split(r"[=<>]", line)[0].strip().lower()
                            if pkg:
                                imported.add(pkg)
            except Exception:
                pass

        return imported

    @classmethod
    def probe(cls, workspace_root: str, target_files: Optional[List[str]] = None) -> EnvironmentInventory:
        """Run complete environment inspection and cross-reference dependencies."""
        installed = cls.get_installed_packages()
        binaries = cls.get_system_binaries()
        gpu_ok, gpu_msg = cls.check_gpu()
        local_wheels = cls.find_local_wheels(workspace_root)
        required = sorted(list(cls.extract_workspace_imports(workspace_root, target_files)))

        # Cross-reference
        available = []
        missing = []

        for req in required:
            # Check if package is installed or specifiable
            req_normalized = req.replace("-", "_").lower()
            installed_normalized = {k.replace("-", "_").lower(): v for k, v in installed.items()}

            if req_normalized in installed_normalized or importlib.util.find_spec(req) is not None:
                available.append(req)
            else:
                missing.append(req)

        return EnvironmentInventory(
            python_version=sys.version.split()[0],
            python_path=sys.executable,
            installed_packages=installed,
            system_binaries=binaries,
            gpu_available=gpu_ok,
            gpu_info=gpu_msg,
            required_packages=required,
            available_packages=sorted(available),
            missing_packages=sorted(missing),
            local_wheels=local_wheels,
        )


class EnvironmentDiscoveryTool(BaseTool):
    """Tool that allows agent to actively explore the environment and dependencies."""

    name = "explore_environment"
    description = (
        "Autonomously scan the sandbox environment to discover installed packages, "
        "available system binaries, GPU hardware, and cross-reference required dependencies."
    )
    parameters = {
        "type": "object",
        "properties": {
            "check_wheels": {"type": "boolean", "description": "Check for local wheel files in workspace"}
        },
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        inventory = EnvironmentProbe.probe(context.workspace_root)

        output_parts = [
            f"=== Sandbox Environment Discovery ===",
            f"Python Runtime: {inventory.python_version} ({inventory.python_path})",
            f"Hardware Accelerator: {inventory.gpu_info}",
            f"Installed Packages Count: {len(inventory.installed_packages)}",
            f"Available System Binaries: {', '.join(sorted(inventory.system_binaries.keys()))}",
        ]

        if inventory.required_packages:
            output_parts.append(f"\nCodebase Required Dependencies ({len(inventory.required_packages)}):")
            for req in inventory.required_packages:
                status = "INSTALLED / AVAILABLE" if req in inventory.available_packages else "MISSING"
                output_parts.append(f"  [{status}] {req}")

        if inventory.local_wheels:
            output_parts.append(f"\nOffline Wheels Found in Workspace ({len(inventory.local_wheels)}):")
            for w in inventory.local_wheels:
                output_parts.append(f"  * {w}")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata=inventory.to_dict(),
        )

