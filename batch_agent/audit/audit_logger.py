"""
Enterprise-grade Audit Logger writing comprehensive audit trails to .txt files.
Flushes immediately to disk to survive preemption in non-interactive batch environments.
"""

from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from batch_agent.audit.events import AuditEvent, AuditEventType, ProcessSnapshot
from batch_agent.audit.formatter import AuditFormatter
from batch_agent.config import AuditConfig


class AuditLogger:
    """Thread-safe, crash-resilient .txt and .csv audit logger."""

    CSV_HEADERS = [
        "timestamp",
        "step",
        "event_type",
        "title",
        "status",
        "exit_code",
        "duration_seconds",
        "command",
        "tokens_used",
        "active_pids",
        "error_summary",
        "description",
    ]

    def __init__(self, config: AuditConfig, workspace_root: str, job_id: str = "") -> None:
        self.config = config
        self.workspace_root = workspace_root
        self.job_id = job_id or f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.log_path = Path(self.config.resolve_path(self.workspace_root))
        self.csv_path = Path(self.config.resolve_csv_path(self.workspace_root)) if self.config.enable_csv_log else None
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []
        self._file_handle = None
        self._csv_handle = None
        self._csv_writer = None

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self.log_path, "a", encoding="utf-8")

        # Initialize CSV log if enabled
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self._csv_handle = open(self.csv_path, "a", encoding="utf-8", newline="")
            self._csv_writer = csv.writer(self._csv_handle)
            if needs_header:
                self._csv_writer.writerow(self.CSV_HEADERS)
                self._csv_handle.flush()
                try:
                    os.fsync(self._csv_handle.fileno())
                except OSError:
                    pass

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.flush()
                try:
                    os.fsync(self._file_handle.fileno())
                except OSError:
                    pass
                self._file_handle.close()

            if self._csv_handle and not self._csv_handle.closed:
                self._csv_handle.flush()
                try:
                    os.fsync(self._csv_handle.fileno())
                except OSError:
                    pass
                self._csv_handle.close()

    def _write_text(self, text: str) -> None:
        if not self._file_handle or self._file_handle.closed:
            self._file_handle = open(self.log_path, "a", encoding="utf-8")

        self._file_handle.write(text + "\n")
        if self.config.flush_immediately:
            self._file_handle.flush()
            try:
                os.fsync(self._file_handle.fileno())
            except OSError:
                pass

    def record_event(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)
            formatted = AuditFormatter.format_event(event)
            self._write_text(formatted)

            # Record to CSV if enabled
            if self._csv_writer and self._csv_handle and not self._csv_handle.closed:
                err_summary = (
                    event.metadata.get("error")
                    or event.security_violation
                    or (event.stderr.splitlines()[0] if event.stderr else "")
                )
                status_str = event.status or (
                    "FAILED" if (event.exit_code not in (None, 0) or event.security_violation) else "SUCCESS"
                )
                dur_str = f"{event.duration_seconds:.3f}" if event.duration_seconds is not None else ""
                row = [
                    event.timestamp,
                    str(event.step_number if event.step_number is not None else ""),
                    event.event_type.value,
                    event.title,
                    status_str,
                    str(event.exit_code if event.exit_code is not None else ""),
                    dur_str,
                    (event.command or "").replace("\n", " ")[:200],
                    str(event.tokens_used or 0),
                    str(len(event.active_processes)),
                    str(err_summary)[:150].replace("\n", " "),
                    event.description.replace("\n", " ")[:250],
                ]
                self._csv_writer.writerow(row)
                if self.config.flush_immediately:
                    self._csv_handle.flush()
                    try:
                        os.fsync(self._csv_handle.fileno())
                    except OSError:
                        pass

    def get_events(self) -> List[AuditEvent]:
        with self._lock:
            return list(self._events)

    def log_job_start(
        self,
        goal: str,
        associated_files: List[str],
        sandbox_profile: str,
        max_steps: int,
        provider: str,
    ) -> None:
        banner = AuditFormatter.banner(f"YOYO - AUDIT TRAIL [JOB: {self.job_id}]")
        now_iso = datetime.now(timezone.utc).isoformat()
        header = (
            f"\n{banner}\n"
            f"Job ID             : {self.job_id}\n"
            f"Execution Mode     : NON-INTERACTIVE BATCH OS ENVIRONMENT\n"
            f"Start Timestamp    : {now_iso}\n"
            f"Workspace Root     : {self.workspace_root}\n"
            f"Audit Log File     : {self.log_path}\n"
            f"Sandbox Boundary   : ENFORCED (Profile: {sandbox_profile.upper()})\n"
            f"Intelligence Model : {provider}\n"
            f"Goal Specification : {goal}\n"
            f"Associated Files   : {', '.join(associated_files) if associated_files else '[None provided]'}\n"
            f"Max Allowed Steps  : {max_steps}\n"
            f"Safety Disclaimer  : YoYo can make mistakes. Always run on copies and maintain backups.\n"
            f"Liability Notice   : The author is not liable for any damages or losses arising from execution.\n"
            f"{'=' * AuditFormatter.LINE_WIDTH}\n"
        )
        with self._lock:
            self._write_text(header)

        self.record_event(
            AuditEvent(
                event_type=AuditEventType.JOB_START,
                title="Batch Job Ingestion & Environment Initialization",
                description="Ingested goal and associated files into non-interactive batch sandbox.",
                metadata={
                    "job_id": self.job_id,
                    "goal": goal,
                    "associated_files": associated_files,
                    "sandbox_profile": sandbox_profile,
                    "provider": provider,
                },
            )
        )

    def log_chain_of_thought(
        self,
        step: int,
        phase: str,
        reasoning: str,
        plan_summary: Optional[str] = None,
        tradeoffs: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> None:
        body = f"Phase: {phase.upper()}\n\n{reasoning}"
        if plan_summary:
            body += f"\n\nCurrent Plan:\n{plan_summary}"
        if tradeoffs:
            body += f"\n\nTrade-offs Considered:\n{tradeoffs}"
        if decision:
            body += f"\n\nOptimal Decision Taken:\n{decision}"

        self.record_event(
            AuditEvent(
                event_type=AuditEventType.CHAIN_OF_THOUGHT,
                step_number=step,
                title=f"Autonomous Reasoning ({phase.capitalize()})",
                chain_of_thought=body,
                metadata={"phase": phase},
            )
        )

    def log_tool_invocation(
        self,
        step: int,
        tool_name: str,
        parameters: Dict[str, Any],
        sandbox_passed: bool,
        result_summary: str,
        error: Optional[str] = None,
    ) -> None:
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.TOOL_INVOCATION,
                step_number=step,
                title=f"Tool Execution: {tool_name}",
                description=f"Invoked tool '{tool_name}' (Sandbox: {'PASSED' if sandbox_passed else 'BLOCKED'}).",
                metadata={
                    "tool": tool_name,
                    "parameters": parameters,
                    "sandbox_status": "ALLOW" if sandbox_passed else "DENY",
                    "result_summary": result_summary,
                    "error": error or "None",
                },
            )
        )

    def log_command_execution(
        self,
        step: int,
        command: str,
        working_dir: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_seconds: float,
        active_processes: List[ProcessSnapshot],
        sanitized_env_keys: List[str],
    ) -> None:
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.COMMAND_EXECUTION,
                step_number=step,
                title=f"Shell Execution: {command[:40]}...",
                description=f"Command executed in {working_dir} (Exit: {exit_code}).",
                command=command,
                exit_code=exit_code,
                stdout=stdout[:4000] if stdout else "",
                stderr=stderr[:2000] if stderr else "",
                duration_seconds=duration_seconds,
                active_processes=active_processes,
                metadata={
                    "working_directory": working_dir,
                    "sanitized_env_keys_stripped": sanitized_env_keys,
                },
            )
        )

    def log_process_event(
        self,
        action: str,
        pid: int,
        command: str,
        active_processes: List[ProcessSnapshot],
        details: str = "",
    ) -> None:
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.PROCESS_EVENT,
                title=f"Process Lifecycle: {action} (PID {pid})",
                description=details or f"Subprocess PID {pid} event: {action}",
                active_processes=active_processes,
                metadata={
                    "action": action,
                    "pid": pid,
                    "command": command,
                },
            )
        )

    def log_security_alert(
        self,
        step: int | None,
        violation: str,
        attempted_target: str,
        policy_rule: str,
    ) -> None:
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.SECURITY_ALERT,
                step_number=step,
                title="SANDBOX SECURITY INTERCEPTION",
                description=f"Blocked prohibited operation violating '{policy_rule}'.",
                security_violation=violation,
                metadata={
                    "attempted_target": attempted_target,
                    "policy_rule": policy_rule,
                },
            )
        )

    def log_verification(
        self,
        step: int,
        check_name: str,
        passed: bool,
        details: str,
    ) -> None:
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.VERIFICATION,
                step_number=step,
                title=f"Verification Check: {check_name} -> {'PASS' if passed else 'FAIL'}",
                description=details,
                metadata={"verification": check_name, "passed": passed},
            )
        )

    def log_self_correction(
        self,
        step: int,
        diagnosis: str,
        remedy_action: str,
        attempt_number: int,
    ) -> None:
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.SELF_CORRECTION,
                step_number=step,
                title=f"Autonomous Self-Correction (Attempt {attempt_number})",
                description=f"Failure detected. Diagnosing root cause and executing corrective action.",
                metadata={
                    "diagnosis": diagnosis,
                    "remedy": remedy_action,
                    "attempt": attempt_number,
                },
            )
        )

    def log_checkpoint(
        self,
        step: int,
        completed_steps: List[str],
        remaining_steps: List[str],
        active_processes: List[ProcessSnapshot],
    ) -> None:
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.CHECKPOINT,
                step_number=step,
                title="Execution Checkpoint State",
                description=f"Completed {len(completed_steps)} subgoals, {len(remaining_steps)} remaining.",
                active_processes=active_processes,
                metadata={
                    "completed_subgoals": completed_steps,
                    "remaining_subgoals": remaining_steps,
                },
            )
        )

    def log_job_completion(
        self,
        success: bool,
        total_steps: int,
        total_runtime_seconds: float,
        final_summary: str,
        artifacts_produced: List[str],
        active_processes: List[ProcessSnapshot],
    ) -> None:
        status_str = "SUCCESS" if success else "FAILED"
        self.record_event(
            AuditEvent(
                event_type=AuditEventType.JOB_COMPLETION,
                title=f"Batch Job Finished: {status_str}",
                description=final_summary,
                active_processes=active_processes,
                metadata={
                    "status": status_str,
                    "total_steps": total_steps,
                    "total_runtime_seconds": total_runtime_seconds,
                    "artifacts": artifacts_produced,
                },
            )
        )

        banner = AuditFormatter.banner("AUDIT TRAIL COMPLETE - FINAL SUMMARY")
        footer = (
            f"\n{banner}\n"
            f"Final Job Status   : {status_str}\n"
            f"Total Steps Taken  : {total_steps}\n"
            f"Total Job Runtime  : {total_runtime_seconds:.2f} seconds\n"
            f"Produced Artifacts : {', '.join(artifacts_produced) if artifacts_produced else '[None]'}\n"
            f"Summary            : {final_summary}\n"
            f"Active Subprocesses: {len(active_processes)} remaining (Cleaned up by YoYo)\n"
            f"{'=' * AuditFormatter.LINE_WIDTH}\n"
        )
        with self._lock:
            self._write_text(footer)

