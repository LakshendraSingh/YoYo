"""
Dynamic Plan Update Tool.
Enables autonomous agents to modify, add, replace, or skip plan steps
to keep the primary goal as top priority in each run and adapt to environment conditions.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from batch_agent.audit.events import AuditEvent, AuditEventType
from batch_agent.tools.base import BaseTool, ToolContext, ToolResult


class UpdatePlanTool(BaseTool):
    """Tool allowing autonomous agent to adapt and mutate its execution plan."""

    name = "update_plan"
    description = (
        "Dynamically alter, add, replace, or skip execution plan steps to ensure the primary goal "
        "is accomplished. Use when blocked, when an error occurs, or when an alternative approach is required."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add_step", "insert_step", "modify_step", "skip_step", "replace_remaining"],
                "description": "Plan mutation operation to execute.",
            },
            "reason": {
                "type": "string",
                "description": "Detailed justification for why the plan is being altered and how it achieves the primary goal.",
            },
            "title": {
                "type": "string",
                "description": "Title of the step (for add_step, insert_step, modify_step).",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the step action.",
            },
            "action_type": {
                "type": "string",
                "enum": ["inspect", "execute", "verify", "report", "patch"],
                "description": "Category of step action.",
            },
            "tool_hint": {
                "type": "string",
                "description": "Optional suggested tool name for this step.",
            },
            "step_id": {
                "type": "integer",
                "description": "Target step ID for modify_step or skip_step.",
            },
            "insert_at": {
                "type": "integer",
                "description": "Zero-based index at which to insert new step (for insert_step).",
            },
            "new_steps": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of step definitions for replace_remaining action.",
            },
        },
        "required": ["action", "reason"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        plan = context.plan
        if not plan:
            return ToolResult(
                success=False,
                error="No active ExecutionPlan found in tool context.",
            )

        action = params.get("action", "")
        reason = params.get("reason", "")
        if not reason:
            return ToolResult(
                success=False,
                error="Parameter 'reason' is required: explain why this plan change accomplishes the primary goal.",
            )

        # Check if Goal Supremacy is enabled or if Strict Plan mode is enforced
        engine = getattr(context, "engine", None)
        if engine and hasattr(engine, "config"):
            if not getattr(engine.config, "goal_supremacy", True):
                return ToolResult(
                    success=False,
                    error="Plan mutation rejected: Goal Supremacy toggle is disabled (strict plan mode). The execution plan cannot be modified.",
                )

        mutation_details: Dict[str, Any] = {"action": action, "reason": reason}

        if action == "add_step":
            title = params.get("title", "Dynamic Subgoal")
            desc = params.get("description", title)
            act_type = params.get("action_type", "execute")
            hint = params.get("tool_hint")
            new_s = plan.add_step(title=title, description=desc, action_type=act_type, tool_hint=hint)
            mutation_details["added_step"] = new_s.to_dict()
            msg = f"Appended new step {new_s.id}: '{title}' to plan."

        elif action == "insert_step":
            title = params.get("title", "Urgent Corrective Subgoal")
            desc = params.get("description", title)
            act_type = params.get("action_type", "execute")
            hint = params.get("tool_hint")
            insert_idx = params.get("insert_at", plan.current_step_index)
            new_s = plan.add_step(
                title=title,
                description=desc,
                action_type=act_type,
                tool_hint=hint,
                insert_at=insert_idx,
            )
            mutation_details["inserted_step"] = new_s.to_dict()
            mutation_details["insert_index"] = insert_idx
            msg = f"Inserted step {new_s.id}: '{title}' at position {insert_idx}."

        elif action == "modify_step":
            step_id = params.get("step_id")
            if step_id is None:
                return ToolResult(success=False, error="Parameter 'step_id' is required for modify_step.")
            ok = plan.modify_step(
                step_id=step_id,
                title=params.get("title"),
                description=params.get("description"),
                action_type=params.get("action_type"),
                tool_hint=params.get("tool_hint"),
            )
            if not ok:
                return ToolResult(success=False, error=f"Step with id {step_id} not found in plan.")
            mutation_details["modified_step_id"] = step_id
            msg = f"Successfully updated step {step_id}."

        elif action == "skip_step":
            step_id = params.get("step_id")
            if step_id is None:
                return ToolResult(success=False, error="Parameter 'step_id' is required for skip_step.")
            ok = plan.skip_step(step_id, reason)
            if not ok:
                return ToolResult(success=False, error=f"Step with id {step_id} not found in plan.")
            mutation_details["skipped_step_id"] = step_id
            msg = f"Marked step {step_id} as SKIPPED. Reason: {reason}"

        elif action == "replace_remaining":
            new_specs = params.get("new_steps", [])
            if not isinstance(new_specs, list):
                return ToolResult(success=False, error="'new_steps' must be a list of step specifications.")
            plan.replace_remaining_steps(new_specs, reason)
            mutation_details["new_steps_count"] = len(new_specs)
            msg = f"Replaced all remaining steps with {len(new_specs)} new steps to achieve goal."

        else:
            return ToolResult(success=False, error=f"Unknown plan mutation action: '{action}'")

        # Record mutation in plan
        plan.record_mutation(
            mutation_type=f"TOOL_{action.upper()}",
            reason=reason,
            details=mutation_details,
        )

        # Sync to plan.txt if it exists in workspace
        plan_path = os.path.join(context.workspace_root, "plan.txt")
        if os.path.exists(plan_path):
            try:
                from batch_agent.intelligence.dependency_planner import DependencyPlanner
                DependencyPlanner.update_plan_file(
                    plan_path=plan_path,
                    plan=plan,
                    mutation_reason=f"[{action.upper()}] {reason}",
                )
                mutation_details["plan_txt_updated"] = True
            except Exception as e:
                mutation_details["plan_txt_update_error"] = str(e)

        # Record structured audit event
        if context.audit_logger:
            context.audit_logger.record_event(
                AuditEvent(
                    event_type=AuditEventType.PLAN_MUTATION,
                    step_number=context.step_number,
                    title=f"Autonomous Plan Mutation ({action})",
                    description=(
                        f"The agent adapted the execution plan to maintain primary goal priority. "
                        f"Action: {action}. Reason: {reason}"
                    ),
                    metadata={
                        "action": action,
                        "reason": reason,
                        "goal": plan.goal,
                        "total_steps": len(plan.steps),
                        "current_step_index": plan.current_step_index,
                        "details": mutation_details,
                    },
                )
            )

        return ToolResult(
            success=True,
            output=(
                f"{msg}\n"
                f"Goal Priority: {plan.goal}\n"
                f"Current Plan Status:\n{plan.format_summary()}"
            ),
            metadata=mutation_details,
        )

