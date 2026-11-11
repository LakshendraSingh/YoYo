"""
Process registry and lifecycle tracker for batch operating system environments.
Monitors active subprocesses, resource metadata, and prevents orphaned child processes.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from batch_agent.audit.events import ProcessSnapshot


@dataclass
class ProcessRecord:
    """Internal tracking record for a spawned subprocess."""
    pid: int
    command: str
    process: Optional[subprocess.Popen]
    start_time: float
    start_time_iso: str
    status: str = "RUNNING"  # RUNNING, COMPLETED, FAILED, TIMED_OUT, TERMINATED
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    timeout_seconds: Optional[int] = None
    stdout_preview: str = ""
    stderr_preview: str = ""

    @property
    def duration_seconds(self) -> float:
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        return max(0.0, time.time() - self.start_time)

    def to_snapshot(self) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=self.pid,
            command=self.command,
            status=self.status,
            start_time=self.start_time_iso,
            duration_seconds=self.duration_seconds,
            exit_code=self.exit_code,
        )


class ProcessTracker:
    """Thread-safe registry for all child processes spawned during the batch job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[int, ProcessRecord] = {}

    def register(
        self,
        process: subprocess.Popen,
        command: str,
        timeout_seconds: Optional[int] = None,
    ) -> ProcessRecord:
        with self._lock:
            now = time.time()
            now_iso = datetime.now(timezone.utc).isoformat()
            record = ProcessRecord(
                pid=process.pid,
                command=command,
                process=process,
                start_time=now,
                start_time_iso=now_iso,
                status="RUNNING",
                timeout_seconds=timeout_seconds,
            )
            self._records[process.pid] = record
            return record

    def update_finished(
        self,
        pid: int,
        exit_code: int,
        timed_out: bool = False,
        terminated: bool = False,
        stdout_preview: str = "",
        stderr_preview: str = "",
    ) -> Optional[ProcessRecord]:
        with self._lock:
            record = self._records.get(pid)
            if not record:
                return None

            record.end_time = time.time()
            record.exit_code = exit_code
            if timed_out:
                record.status = "TIMED_OUT"
            elif terminated:
                record.status = "TERMINATED"
            elif exit_code == 0:
                record.status = "COMPLETED"
            else:
                record.status = "FAILED"

            record.stdout_preview = stdout_preview[:500]
            record.stderr_preview = stderr_preview[:500]
            return record

    def get_active_snapshots(self) -> List[ProcessSnapshot]:
        """Poll and return a snapshot of all currently running child processes."""
        with self._lock:
            snapshots = []
            for record in self._records.values():
                if record.status == "RUNNING":
                    # Check if process actually exited behind the scenes
                    if record.process is not None:
                        poll_result = record.process.poll()
                        if poll_result is not None:
                            record.end_time = time.time()
                            record.exit_code = poll_result
                            record.status = "COMPLETED" if poll_result == 0 else "FAILED"
                            continue
                    snapshots.append(record.to_snapshot())
            return snapshots

    def get_all_records(self) -> List[ProcessRecord]:
        with self._lock:
            return list(self._records.values())

    def terminate_all(self, grace_period_seconds: float = 2.0) -> int:
        """Gracefully terminate then force-kill all active tracked child processes."""
        with self._lock:
            active = [r for r in self._records.values() if r.status == "RUNNING" and r.process]

        if not active:
            return 0

        # Phase 1: SIGTERM
        for r in active:
            try:
                if r.process and r.process.poll() is None:
                    r.process.terminate()
            except OSError:
                pass

        time.sleep(min(0.5, grace_period_seconds))

        # Phase 2: SIGKILL for survivors
        killed_count = 0
        for r in active:
            try:
                if r.process and r.process.poll() is None:
                    r.process.kill()
                    r.process.wait(timeout=1.0)
                    r.status = "TERMINATED"
                    killed_count += 1
                else:
                    r.status = "TERMINATED"
            except Exception:
                pass

        return len(active)

