"""
Tool for inspecting and controlling child processes in the batch execution environment.
"""

from __future__ import annotations

import os
import signal
from typing import Any, Dict

from batch_agent.audit.formatter import AuditFormatter
from batch_agent.tools.base import BaseTool, ToolContext, ToolResult


class ProcessInspectTool(BaseTool):
    """Tool to inspect active subprocesses or terminate runaway PIDs."""

    name = "process_inspect"
    description = (
        "Inspect active child subprocesses, query execution durations, "
        "or terminate runaway processes by PID."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "terminate"],
                "description": "Action to perform: 'list' active processes or 'terminate' a PID",
            },
            "pid": {"type": "integer", "description": "Process ID to terminate (if action='terminate')"},
        },
        "required": ["action"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        action = params.get("action", "list")

        if action == "list":
            active_snapshots = context.tracker.get_active_snapshots()
            formatted = AuditFormatter.format_process_list(active_snapshots)
            return ToolResult(
                success=True,
                output=f"Active Subprocesses:\n{formatted}",
                metadata={"active_count": len(active_snapshots)},
            )

        elif action == "terminate":
            pid = params.get("pid")
            if not pid:
                return ToolResult(success=False, error="PID is required for terminate action")

            # Verify PID is tracked
            active = {p.pid: p for p in context.tracker.get_active_snapshots()}
            if pid not in active:
                return ToolResult(success=False, error=f"PID {pid} is not an active tracked child process")

            try:
                os.kill(pid, signal.SIGTERM)
                return ToolResult(
                    success=True,
                    output=f"Sent SIGTERM to process PID {pid}.",
                    metadata={"terminated_pid": pid},
                )
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to terminate PID {pid}: {e}")

        return ToolResult(success=False, error=f"Unknown action: {action}")

