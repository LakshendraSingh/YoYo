"""
State representation for batch execution jobs with crash-resilient checkpointing.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from batch_agent.intelligence.planner import ExecutionPlan


@dataclass
class BatchJobState:
    """Live state maintained throughout the batch execution lifecycle."""
    job_id: str
    goal: str
    workspace_root: str
    plan: ExecutionPlan
    current_step_number: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    is_success: bool = False
    exit_code: int = 0
    error_message: Optional[str] = None
    artifacts_produced: List[str] = field(default_factory=list)
    last_observation: str = ""
    last_error: Optional[str] = None
    cumulative_tokens_used: int = 0

    @property
    def elapsed_seconds(self) -> float:
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        return max(0.0, time.time() - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "goal": self.goal,
            "workspace_root": self.workspace_root,
            "current_step_number": self.current_step_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_success": self.is_success,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "artifacts_produced": self.artifacts_produced,
            "last_observation": self.last_observation,
            "last_error": self.last_error,
            "cumulative_tokens_used": self.cumulative_tokens_used,
            "plan": self.plan.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BatchJobState:
        plan = ExecutionPlan.from_dict(data["plan"])
        state = cls(
            job_id=data["job_id"],
            goal=data["goal"],
            workspace_root=data["workspace_root"],
            plan=plan,
            current_step_number=data.get("current_step_number", 0),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            is_success=data.get("is_success", False),
            exit_code=data.get("exit_code", 0),
            error_message=data.get("error_message"),
            artifacts_produced=data.get("artifacts_produced", []),
            last_observation=data.get("last_observation", ""),
            last_error=data.get("last_error"),
            cumulative_tokens_used=data.get("cumulative_tokens_used", 0),
        )
        return state

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Persist state checkpoint to disk synchronously."""
        target_path = path or os.path.join(self.workspace_root, ".batch_checkpoint.json")
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target_path)
        return target_path

    @classmethod
    def load_checkpoint(cls, path: str) -> Optional[BatchJobState]:
        """Load state checkpoint from disk."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            return None

