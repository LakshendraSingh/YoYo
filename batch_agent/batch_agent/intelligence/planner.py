"""
Planning abstractions, dynamic replanning data structures, and goal-priority tracking
for autonomous batch execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class StepStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class PlanStep:
    """A discrete subgoal step within the overall batch execution plan."""
    id: int
    title: str
    description: str
    action_type: str  # inspect, edit, command, verify, report, patch
    tool_hint: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result_summary: str = ""
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def mark_in_progress(self) -> None:
        self.status = StepStatus.IN_PROGRESS

    def mark_completed(self, summary: str) -> None:
        self.status = StepStatus.COMPLETED
        self.result_summary = summary

    def mark_failed(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error_message = error
        self.retry_count += 1

    def mark_skipped(self, reason: str = "") -> None:
        self.status = StepStatus.SKIPPED
        self.result_summary = f"Skipped: {reason}" if reason else "Skipped by agent"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "action_type": self.action_type,
            "tool_hint": self.tool_hint,
            "status": self.status.value,
            "result_summary": self.result_summary,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanStep:
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            action_type=data.get("action_type", "execute"),
            tool_hint=data.get("tool_hint"),
            status=StepStatus(data.get("status", "PENDING")),
            result_summary=data.get("result_summary", ""),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )


@dataclass
class ExecutionPlan:
    """
    Full execution plan consisting of sequential and dynamic subgoals.
    Keeps the PRIMARY GOAL as top priority: allows dynamic mutation, step insertion,
    re-planning, and step replacement whenever needed to accomplish the goal.
    """
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    plan_mutations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def primary_goal(self) -> str:
        return self.goal

    @primary_goal.setter
    def primary_goal(self, value: str) -> None:
        self.goal = value

    def add_step(
        self,
        title: str,
        description: str,
        action_type: str,
        tool_hint: Optional[str] = None,
        insert_at: Optional[int] = None,
    ) -> PlanStep:
        """Add or insert a new subgoal into the execution plan."""
        step_id = len(self.steps) + 1
        step = PlanStep(
            id=step_id,
            title=title,
            description=description,
            action_type=action_type,
            tool_hint=tool_hint,
        )
        if insert_at is not None and 0 <= insert_at <= len(self.steps):
            self.steps.insert(insert_at, step)
            # Re-index step IDs
            for idx, s in enumerate(self.steps, start=1):
                s.id = idx
        else:
            self.steps.append(step)
        return step

    def modify_step(
        self,
        step_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        action_type: Optional[str] = None,
        tool_hint: Optional[str] = None,
    ) -> bool:
        """Dynamically update an existing step's specification."""
        for s in self.steps:
            if s.id == step_id:
                if title is not None:
                    s.title = title
                if description is not None:
                    s.description = description
                if action_type is not None:
                    s.action_type = action_type
                if tool_hint is not None:
                    s.tool_hint = tool_hint
                return True
        return False

    def skip_step(self, step_id: int, reason: str = "") -> bool:
        """Mark a step as skipped if rendered unnecessary for goal completion."""
        for s in self.steps:
            if s.id == step_id:
                s.mark_skipped(reason)
                self.record_mutation(
                    mutation_type="SKIP_STEP",
                    reason=reason or f"Skipped step {step_id}",
                    details={"step_id": step_id, "title": s.title},
                )
                return True
        return False

    def replace_remaining_steps(self, new_step_specs: List[Dict[str, Any]], reason: str) -> None:
        """
        Dynamically replace all pending/uncompleted steps with a new sequence of actions
        tailored to accomplish the primary goal under updated environment conditions.
        """
        completed = [s for s in self.steps if s.status == StepStatus.COMPLETED]
        new_steps: List[PlanStep] = list(completed)

        for spec in new_step_specs:
            step_id = len(new_steps) + 1
            step = PlanStep(
                id=step_id,
                title=spec.get("title", f"Subgoal {step_id}"),
                description=spec.get("description", ""),
                action_type=spec.get("action_type", "execute"),
                tool_hint=spec.get("tool_hint"),
                status=StepStatus.PENDING,
            )
            new_steps.append(step)

        old_count = len(self.steps)
        self.steps = new_steps
        self.current_step_index = len(completed)

        self.record_mutation(
            mutation_type="REPLACE_REMAINING",
            reason=reason,
            details={
                "previous_total": old_count,
                "completed_retained": len(completed),
                "new_total": len(self.steps),
            },
        )

    def record_mutation(self, mutation_type: str, reason: str, details: Dict[str, Any]) -> None:
        """Record plan change event for audit trail and plan.txt sync."""
        self.plan_mutations.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": mutation_type,
            "reason": reason,
            "details": details,
        })

    @property
    def current_step(self) -> Optional[PlanStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_finished(self) -> bool:
        if not self.steps:
            return False
        return self.current_step_index >= len(self.steps)

    @property
    def all_completed(self) -> bool:
        if not self.steps:
            return False
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps)

    def advance(self) -> Optional[PlanStep]:
        self.current_step_index += 1
        return self.current_step

    def format_summary(self) -> str:
        lines = [f"Primary Goal Priority: {self.goal}"]
        for s in self.steps:
            marker = {
                StepStatus.PENDING: "[ ]",
                StepStatus.IN_PROGRESS: "[>]",
                StepStatus.COMPLETED: "[x]",
                StepStatus.FAILED: "[!]",
                StepStatus.SKIPPED: "[-]",
            }.get(s.status, "[?]")
            lines.append(f"{marker} Step {s.id}: {s.title} ({s.status.value})")
        if self.plan_mutations:
            lines.append(f"(Plan has been dynamically adapted {len(self.plan_mutations)} times to serve goal)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "primary_goal": self.goal,
            "current_step_index": self.current_step_index,
            "steps": [s.to_dict() for s in self.steps],
            "plan_mutations": self.plan_mutations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionPlan:
        plan = cls(
            goal=data.get("goal") or data.get("primary_goal", ""),
            current_step_index=data.get("current_step_index", 0),
            plan_mutations=data.get("plan_mutations", []),
        )
        plan.steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return plan
