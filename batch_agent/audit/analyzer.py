"""
Audit Trail Analyzer and Verification utility.
Parses .txt audit log files to produce operational health metrics, timeline summaries,
and verification dashboards for cluster administrators and researchers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AuditReportSummary:
    """Aggregated operational metrics parsed from an audit trail text file."""
    job_id: str = "unknown"
    start_time: str = ""
    goal: str = ""
    sandbox_profile: str = ""
    final_status: str = "UNKNOWN"
    total_steps: int = 0
    total_runtime_seconds: float = 0.0
    phases_count: Dict[str, int] = field(default_factory=dict)
    tools_invoked: Dict[str, int] = field(default_factory=dict)
    commands_executed: List[Dict[str, Any]] = field(default_factory=list)
    security_alerts_count: int = 0
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

    def format_dashboard(self) -> str:
        lines = [
            "=" * 78,
            f"  YOYO - AUDIT LOG ANALYSIS REPORT",
            "=" * 78,
            f"Job ID             : {self.job_id}",
            f"Final Job Status   : {self.final_status}",
            f"Goal               : {self.goal}",
            f"Total Steps        : {self.total_steps}",
            f"Total Runtime      : {self.total_runtime_seconds:.2f}s",
            f"Sandbox Profile    : {self.sandbox_profile}",
            f"Security Alerts    : {self.security_alerts_count}",
            "-" * 78,
            "Tool Invocations Summary:",
        ]
        if self.tools_invoked:
            for tool, count in sorted(self.tools_invoked.items()):
                lines.append(f"  - {tool:<20}: {count} calls")
        else:
            lines.append("  [No tools recorded]")

        lines.append("-" * 78)
        lines.append("Verification Checks:")
        if self.verification_checks:
            for v in self.verification_checks:
                status = "PASS" if v.get("passed") else "FAIL"
                lines.append(f"  [{status}] {v.get('name', 'Check')}")
        else:
            lines.append("  [No verification checks recorded]")

        lines.append("-" * 78)
        lines.append("Commands Executed:")
        if self.commands_executed:
            for cmd in self.commands_executed[:10]:
                lines.append(f"  Exit {cmd.get('exit_code', '?')}: {cmd.get('command', '')[:55]}")
            if len(self.commands_executed) > 10:
                lines.append(f"  ... and {len(self.commands_executed) - 10} more commands")
        else:
            lines.append("  [No external commands recorded]")

        lines.append("-" * 78)
        lines.append(f"Produced Artifacts: {', '.join(self.artifacts) if self.artifacts else '[None]'}")
        lines.append("=" * 78)
        return "\n".join(lines)


class AuditAnalyzer:
    """Parses and verifies structured YoYo .txt audit files."""

    @classmethod
    def parse_file(cls, log_path: str) -> AuditReportSummary:
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Audit log file not found: {log_path}")

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        summary = AuditReportSummary()

        # Extract Job ID
        job_match = re.search(r"Job ID\s*:\s*([^\n]+)", content)
        if job_match:
            summary.job_id = job_match.group(1).strip()

        # Extract Goal
        goal_match = re.search(r"Goal Specification\s*:\s*([^\n]+)", content)
        if goal_match:
            summary.goal = goal_match.group(1).strip()

        # Extract Sandbox Profile
        sb_match = re.search(r"Sandbox Boundary\s*:\s*ENFORCED\s*\(Profile:\s*([^\)]+)\)", content)
        if sb_match:
            summary.sandbox_profile = sb_match.group(1).strip()

        # Extract Final Status
        status_match = re.search(r"Final Job Status\s*:\s*([A-Z]+)", content)
        if status_match:
            summary.final_status = status_match.group(1).strip()

        # Extract Total Steps
        steps_match = re.search(r"Total Steps Taken\s*:\s*(\d+)", content)
        if steps_match:
            summary.total_steps = int(steps_match.group(1).strip())

        # Extract Total Runtime
        runtime_match = re.search(r"Total Job Runtime\s*:\s*([\d\.]+)\s*seconds", content)
        if runtime_match:
            summary.total_runtime_seconds = float(runtime_match.group(1).strip())

        # Extract Produced Artifacts
        artifacts_match = re.search(r"Produced Artifacts\s*:\s*([^\n]+)", content)
        if artifacts_match:
            raw_arts = artifacts_match.group(1).strip()
            if raw_arts and raw_arts != "[None]":
                summary.artifacts = [a.strip() for a in raw_arts.split(",") if a.strip()]

        # Extract Tool Invocations
        tool_matches = re.findall(r"Tool Execution:\s*([a-zA-Z0-9_\-]+)", content)
        for t in tool_matches:
            summary.tools_invoked[t] = summary.tools_invoked.get(t, 0) + 1

        # Extract Verification Checks
        verify_matches = re.findall(r"Verification Check:\s*([^>\n]+)\s*->\s*(PASS|FAIL)", content)
        for name, outcome in verify_matches:
            summary.verification_checks.append({
                "name": name.strip(),
                "passed": outcome == "PASS",
            })

        # Extract Security Alerts
        summary.security_alerts_count = len(re.findall(r"\[SECURITY AUDIT ALERT\]", content))

        # Extract Executed Commands
        cmd_blocks = re.findall(r"Command\s*:\s*([^\n]+)\nExit Code\s*:\s*(\d+)", content)
        for cmd_str, exit_code in cmd_blocks:
            summary.commands_executed.append({
                "command": cmd_str.strip(),
                "exit_code": int(exit_code),
            })

        return summary

