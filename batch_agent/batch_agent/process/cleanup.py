"""
Process cleanup handlers and signal traps for robust batch OS preemption handling.
Ensures zero orphaned child processes when Slurm/PBS/Batch sends SIGTERM or exits.
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from typing import Optional

from batch_agent.audit.audit_logger import AuditLogger
from batch_agent.process.tracker import ProcessTracker

logger = logging.getLogger(__name__)


class BatchProcessCleaner:
    """Installs safety traps to terminate child processes on shutdown or batch preemption."""

    _instance: Optional[BatchProcessCleaner] = None

    def __init__(self, tracker: ProcessTracker, audit_logger: Optional[AuditLogger] = None) -> None:
        self.tracker = tracker
        self.audit_logger = audit_logger
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        atexit.register(self._cleanup_on_exit)

        # Handle POSIX signals gracefully (Slurm preemption / wallclock expiration)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError):
                # Signals might not be assignable in some non-main threads
                pass

        self._installed = True

    def _cleanup_on_exit(self) -> None:
        active = self.tracker.get_active_snapshots()
        if active:
            if self.audit_logger:
                self.audit_logger.log_process_event(
                    action="CLEANUP_ON_EXIT",
                    pid=0,
                    command="[System Shutdown]",
                    active_processes=active,
                    details=f"Terminating {len(active)} surviving background child processes.",
                )
            self.tracker.terminate_all()

    def _signal_handler(self, signum: int, frame: object) -> None:
        signame = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        if self.audit_logger:
            self.audit_logger.log_process_event(
                action="SIGNAL_RECEIVED",
                pid=0,
                command="[OS Signal Trap]",
                active_processes=self.tracker.get_active_snapshots(),
                details=f"Received OS signal {signame}. Initiating immediate graceful cleanup.",
            )
        self.tracker.terminate_all()
        if self.audit_logger:
            self.audit_logger.close()
        sys.exit(128 + signum)

