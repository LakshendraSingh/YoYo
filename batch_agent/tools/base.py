"""
Base abstraction for tools invoked by the autonomous batch agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from batch_agent.audit.audit_logger import AuditLogger
from batch_agent.process.runner import ProcessRunner
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager


@dataclass
class ToolResult:
    """Standardized output structure for tool execution."""
    success: bool
    output: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """Runtime context injected into every tool execution."""
    workspace_root: str
    cwd: str
    sandbox: SandboxManager
    tracker: ProcessTracker
    runner: ProcessRunner
    audit_logger: Optional[AuditLogger] = None
    step_number: Optional[int] = None
    plan: Optional[Any] = None
    engine: Optional[Any] = None


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool logic."""
        pass

