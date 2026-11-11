"""
Text formatting engine for structured, human-readable .txt audit logs.
"""

from __future__ import annotations

import textwrap
from typing import List
from batch_agent.audit.events import AuditEvent, AuditEventType, ProcessSnapshot


class AuditFormatter:
    """Renders structured AuditEvents into clear ASCII/Unicode .txt blocks."""

    LINE_WIDTH = 90
    BANNER_CHAR = "="
    SUB_BANNER_CHAR = "-"

    @classmethod
    def banner(cls, title: str) -> str:
        pad = max(4, (cls.LINE_WIDTH - len(title) - 2) // 2)
        line = cls.BANNER_CHAR * pad + f" {title} " + cls.BANNER_CHAR * pad
        if len(line) < cls.LINE_WIDTH:
            line += cls.BANNER_CHAR * (cls.LINE_WIDTH - len(line))
        return line

    @classmethod
    def section_header(cls, timestamp: str, event_type: str, step: int | None = None) -> str:
        step_part = f" [STEP {step}]" if step is not None else ""
        header = f"--- [{timestamp}] [{event_type}]{step_part} "
        remaining = max(3, cls.LINE_WIDTH - len(header))
        return header + (cls.SUB_BANNER_CHAR * remaining)

    @classmethod
    def format_process_list(cls, processes: List[ProcessSnapshot]) -> str:
        if not processes:
            return "  [No active background child processes]"
        lines = [
            f"  {'PID':<8} {'STATUS':<12} {'DURATION':<10} {'COMMAND'}",
            f"  {'-'*6:<8} {'-'*10:<12} {'-'*8:<10} {'-'*40}"
        ]
        for p in processes:
            dur = f"{p.duration_seconds:.2f}s"
            cmd = p.command if len(p.command) <= 55 else p.command[:52] + "..."
            lines.append(f"  {p.pid:<8} {p.status:<12} {dur:<10} {cmd}")
        return "\n".join(lines)

    @classmethod
    def format_event(cls, event: AuditEvent) -> str:
        out: List[str] = []
        out.append(cls.section_header(event.timestamp, event.event_type.value, event.step_number))

        if event.title:
            out.append(f"Title       : {event.title}")

        if event.description:
            out.append("Description : " + event.description.strip())

        # Metadata attributes
        if event.metadata:
            for k, v in event.metadata.items():
                if isinstance(v, list):
                    if not v:
                        out.append(f"  {k:<16}: []")
                    else:
                        out.append(f"  {k:<16}:")
                        for item in v:
                            out.append(f"    - {item}")
                elif isinstance(v, dict):
                    out.append(f"  {k:<16}:")
                    for subk, subv in v.items():
                        out.append(f"    * {subk}: {subv}")
                else:
                    out.append(f"  {k:<16}: {v}")

        # Chain of Thought
        if event.chain_of_thought:
            out.append("\n[REASONING / CHAIN OF THOUGHT]")
            cot_indented = textwrap.indent(event.chain_of_thought.strip(), "  ")
            out.append(cot_indented)

        # Command Execution specifics
        if event.command:
            out.append(f"\nCommand     : {event.command}")
            if event.exit_code is not None:
                out.append(f"Exit Code   : {event.exit_code}")
            if event.duration_seconds is not None:
                out.append(f"Duration    : {event.duration_seconds:.3f}s")
            if event.stdout:
                out.append("Stdout Output:")
                out.append(textwrap.indent(event.stdout.strip(), "  | "))
            if event.stderr:
                out.append("Stderr Output:")
                out.append(textwrap.indent(event.stderr.strip(), "  ! "))

        # Security Violation
        if event.security_violation:
            out.append("\n[SECURITY AUDIT ALERT]")
            out.append(f"  VIOLATION: {event.security_violation}")

        # Active Processes
        if event.active_processes or event.event_type in (
            AuditEventType.COMMAND_EXECUTION,
            AuditEventType.PROCESS_EVENT,
            AuditEventType.CHECKPOINT,
            AuditEventType.JOB_COMPLETION,
        ):
            out.append("\nActive Subprocesses Snapshot:")
            out.append(cls.format_process_list(event.active_processes))

        out.append("")  # blank trailing line
        return "\n".join(out)

