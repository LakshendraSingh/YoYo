"""
Chain of Thought reasoning generator and structured decision framework.
Captures the mental model, hypotheses, trade-offs, and decisions for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChainOfThoughtRecord:
    """Structured representation of a single agentic decision step."""
    phase: str  # PLANNING, ACTING, REFLECTING, REPLANNING, VERIFYING, MUTATION
    goal_analysis: str
    observation: str
    hypotheses: List[str]
    tradeoffs: List[str]
    decision: str
    expected_outcome: str
    goal_priority: str = ""

    def format_text(self) -> str:
        lines = [
            f"Goal Assessment   : {self.goal_analysis}",
        ]
        if self.goal_priority:
            lines.append(f"Goal Alignment    : {self.goal_priority}")
        lines.append(f"Observation State : {self.observation}")
        if self.hypotheses:
            lines.append("Hypotheses Formed :")
            for h in self.hypotheses:
                lines.append(f"  * {h}")
        if self.tradeoffs:
            lines.append("Trade-offs Evaluated:")
            for t in self.tradeoffs:
                lines.append(f"  * {t}")
        lines.append(f"Optimal Decision  : {self.decision}")
        lines.append(f"Expected Outcome  : {self.expected_outcome}")
        return "\n".join(lines)


class ChainOfThoughtEngine:
    """Helper to formulate and log structured thoughts during batch execution."""

    @classmethod
    def create_thought(
        cls,
        phase: str,
        goal_analysis: str,
        observation: str,
        decision: str,
        hypotheses: Optional[List[str]] = None,
        tradeoffs: Optional[List[str]] = None,
        expected_outcome: str = "",
        goal_priority: str = "",
    ) -> ChainOfThoughtRecord:
        return ChainOfThoughtRecord(
            phase=phase,
            goal_analysis=goal_analysis,
            observation=observation,
            hypotheses=hypotheses or [],
            tradeoffs=tradeoffs or [],
            decision=decision,
            expected_outcome=expected_outcome or "Action advances execution plan towards goal completion.",
            goal_priority=goal_priority,
        )

