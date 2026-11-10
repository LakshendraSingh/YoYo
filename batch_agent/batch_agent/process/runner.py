"""
Subprocess runner engineered for non-interactive batch operating environments.
Enforces timeouts, stdin isolation (DEVNULL), environment scrubbing, and process tracking.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from batch_agent.audit.audit_logger import AuditLogger
from batch_agent.audit.events import ProcessSnapshot
from batch_agent.config import ProcessConfig
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager


@dataclass
class RunResult:
    """Outcome of a shell command execution."""
    command: str = ""
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    sandbox_blocked: bool = False
    block_reason: str = ""
    active_processes: List[ProcessSnapshot] = field(default_factory=list)


# Alias for compatibility
ProcessResult = RunResult


class ProcessRunner:
    """Executes external shell commands safely within batch constraints."""

    def __init__(
        self,
        sandbox: SandboxManager,
        tracker: ProcessTracker,
        audit_logger: Optional[AuditLogger] = None,
        config: Optional[ProcessConfig] = None,
    ) -> None:
        self.sandbox = sandbox
        self.tracker = tracker
        self.audit_logger = audit_logger
        self.config = config or ProcessConfig()

    def run(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        step_number: Optional[int] = None,
        custom_env: Optional[Dict[str, str]] = None,
    ) -> RunResult:
        start_time = time.time()
        timeout = timeout_seconds if timeout_seconds is not None else self.config.default_timeout_seconds
        if timeout is not None and timeout <= 0:
            timeout = None
        working_dir = cwd or self.sandbox.config.workspace_root

        # 1. Sandbox command safety validation
        allowed, reason = self.sandbox.validate_command(command)
        if not allowed:
            if self.audit_logger:
                self.audit_logger.log_security_alert(
                    step=step_number,
                    violation=reason,
                    attempted_target=command,
                    policy_rule="CommandGuard",
                )
            return RunResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=f"Security sandbox blocked command: {reason}",
                duration_seconds=0.0,
                sandbox_blocked=True,
                block_reason=reason,
                active_processes=self.tracker.get_active_snapshots(),
            )

        # 2. Scrub environment
        merged_env = os.environ.copy()
        if custom_env:
            merged_env.update(custom_env)
        sanitized_env, stripped_keys = self.sandbox.sanitize_environment(merged_env)

        # 3. Spawn subprocess with stdin disconnected (DEVNULL)
        proc: Optional[subprocess.Popen] = None
        timed_out = False
        stdout_text = ""
        stderr_text = ""
        exit_code = -1

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.DEVNULL,  # Crucial for non-interactive batch operating systems
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                env=sanitized_env,
                start_new_session=True,  # Isolate process group
            )

            # Register with active tracker
            record = self.tracker.register(proc, command, timeout_seconds=timeout)

            # Log process spawn to audit logger if available
            if self.audit_logger:
                self.audit_logger.log_process_event(
                    action="SPAWNED",
                    pid=proc.pid,
                    command=command,
                    active_processes=self.tracker.get_active_snapshots(),
                    details=f"Spawned PID {proc.pid} in {working_dir} (Timeout: {timeout}s)",
                )

            # Wait for execution completion or timeout
            stdout_raw, stderr_raw = proc.communicate(timeout=timeout)
            stdout_text = stdout_raw or ""
            stderr_text = stderr_raw or ""
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124  # Standard timeout exit code
            stderr_text = f"Execution timed out after {timeout} seconds."
            if proc:
                try:
                    proc.terminate()
                    time.sleep(0.5)
                    if proc.poll() is None:
                        proc.kill()
                    stdout_raw, stderr_raw_extra = proc.communicate()
                    stdout_text = stdout_raw or ""
                    stderr_text += "\n" + (stderr_raw_extra or "")
                except Exception:
                    pass

        except Exception as exc:
            exit_code = 1
            stderr_text = f"Process launch failed: {exc}"

        duration = time.time() - start_time

        # Update tracker
        if proc:
            self.tracker.update_finished(
                pid=proc.pid,
                exit_code=exit_code,
                timed_out=timed_out,
                stdout_preview=stdout_text,
                stderr_preview=stderr_text,
            )

        active_snapshots = self.tracker.get_active_snapshots()

        # Log command execution to audit logger
        if self.audit_logger:
            self.audit_logger.log_command_execution(
                step=step_number or 0,
                command=command,
                working_dir=working_dir,
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_seconds=duration,
                active_processes=active_snapshots,
                sanitized_env_keys=stripped_keys,
            )

        return RunResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            timed_out=timed_out,
            sandbox_blocked=False,
            active_processes=active_snapshots,
        )

