"""
Structured audit event types and models for the batch agent audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AuditEventType(str, Enum):
    """Categories of auditable events."""
    JOB_START = "JOB_START"
    CHAIN_OF_THOUGHT = "CHAIN_OF_THOUGHT"
    TOOL_INVOCATION = "TOOL_INVOCATION"
    COMMAND_EXECUTION = "COMMAND_EXECUTION"
    PROCESS_EVENT = "PROCESS_EVENT"
    SECURITY_ALERT = "SECURITY_ALERT"
    DOCUMENT_INGESTION = "DOCUMENT_INGESTION"
    PLAN_MUTATION = "PLAN_MUTATION"
    VERIFICATION = "VERIFICATION"
    SELF_CORRECTION = "SELF_CORRECTION"
    RECOVERY_ACTION = "RECOVERY_ACTION"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    CHECKPOINT = "CHECKPOINT"
    JOB_COMPLETION = "JOB_COMPLETION"
    ERROR = "ERROR"


@dataclass
class ProcessSnapshot:
    """Snapshot of a process in the execution registry."""
    pid: int
    command: str
    status: str  # RUNNING, COMPLETED, FAILED, TIMED_OUT, TERMINATED
    start_time: str
    duration_seconds: float = 0.0
    exit_code: Optional[int] = None
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None


@dataclass
class AuditEvent:
    """Unified audit trail event record."""
    event_type: AuditEventType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    step_number: Optional[int] = None
    title: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    chain_of_thought: Optional[str] = None
    command: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration_seconds: Optional[float] = None
    active_processes: List[ProcessSnapshot] = field(default_factory=list)
    security_violation: Optional[str] = None
    tokens_used: int = 0
    status: str = ""
    recovery_details: Optional[str] = None

