"""
Shell command execution tool with non-interactive batch guarantees and active process tracking.
"""

from __future__ import annotations

from typing import Any, Dict

from batch_agent.process.recovery import CommandRecoveryEngine
from batch_agent.tools.base import BaseTool, ToolContext, ToolResult


class CommandExecutionTool(BaseTool):
    """Executes a shell command inside the sandbox."""

    name = "run_command"
    description = (
        "Run an external command non-interactively within the sandbox workspace. "
        "Captures stdout, stderr, exit code, execution duration, and tracks active child processes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command line to execute"},
            "cwd": {"type": "string", "description": "Optional working directory"},
            "timeout_seconds": {"type": "integer", "description": "Execution timeout limit in seconds"},
        },
        "required": ["command"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        command = params.get("command", "")
        cwd = params.get("cwd") or context.cwd
        timeout = params.get("timeout_seconds")

        run_res = context.runner.run(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout,
            step_number=context.step_number,
        )

        output_str = run_res.stdout
        if run_res.stderr:
            output_str += ("\n[STDERR]\n" + run_res.stderr) if output_str else run_res.stderr

        active_count = len(run_res.active_processes)
        active_summary = f"[Active Background Processes: {active_count}]"

        combined_output = (
            f"Exit Code: {run_res.exit_code} (Duration: {run_res.duration_seconds:.2f}s)\n"
            f"{active_summary}\n"
            f"{output_str}"
        )

        success = (run_res.exit_code == 0) and not run_res.sandbox_blocked and not run_res.timed_out
        recovery_applied: Optional[str] = None

        # Autonomous Detection and Recovery from Failed Shell Commands
        enable_rec = getattr(context.engine.config, "enable_recovery_engine", True) if hasattr(context, "engine") and context.engine else True
        if not success and not run_res.sandbox_blocked and enable_rec:
            try:
                recovery = CommandRecoveryEngine.attempt_recovery(
                    command=command,
                    result=run_res,
                    context=context,
                    plan=context.plan,
                )
                if recovery.success and recovery.repaired_result:
                    run_res = recovery.repaired_result
                    success = (run_res.exit_code == 0)
                    recovery_applied = recovery.strategy_applied
                    output_str = run_res.stdout
                    if run_res.stderr:
                        output_str += ("\n[STDERR]\n" + run_res.stderr) if output_str else run_res.stderr
                    combined_output = (
                        f"[AUTONOMOUS RECOVERY SUCCESSFUL]\n"
                        f"Strategy Applied : {recovery.strategy_applied}\n"
                        f"Repaired Command : {recovery.repaired_command}\n"
                        f"Exit Code: {run_res.exit_code} (Duration: {run_res.duration_seconds:.2f}s)\n"
                        f"{output_str}"
                    )
            except Exception as e:
                # Never let recovery errors crash the batch tool
                combined_output += f"\n[Recovery Attempt Warning: {e}]"

        error_msg = run_res.block_reason if run_res.sandbox_blocked else (
            "Command timed out" if run_res.timed_out else (
                f"Command failed with exit code {run_res.exit_code}" if not success else None
            )
        )

        return ToolResult(
            success=success,
            output=combined_output,
            error=error_msg,
            metadata={
                "exit_code": run_res.exit_code,
                "duration_seconds": run_res.duration_seconds,
                "timed_out": run_res.timed_out,
                "active_processes_count": active_count,
                "recovery_applied": recovery_applied,
            },
        )

