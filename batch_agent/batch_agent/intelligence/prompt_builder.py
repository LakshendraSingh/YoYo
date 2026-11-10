"""
System prompt generator and context builder tailored for non-interactive batch environments.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


class PromptBuilder:
    """Constructs focused, batch-specialized context prompts."""

    @classmethod
    def build_system_prompt(
        cls,
        workspace_root: str,
        associated_files: List[str],
        tool_schemas: List[Dict[str, Any]],
        sandbox_profile: str,
        document_digests: Optional[List[Any]] = None,
        goal_supremacy: bool = True,
    ) -> str:
        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tool_schemas
        )

        files_desc = "\n".join(f"- {f}" for f in associated_files) if associated_files else "[None specified]"

        docs_block = ""
        if document_digests:
            doc_snippets = []
            for d in document_digests:
                if hasattr(d, "to_prompt_context"):
                    doc_snippets.append(d.to_prompt_context(max_chars=2000))
                else:
                    doc_snippets.append(str(d))
            docs_block = "\n\nINGESTED RESEARCH PAPERS & SRS SPECIFICATIONS:\n" + "\n\n".join(doc_snippets)

        planning_rule = """1. GOAL SUPREMACY & DYNAMIC REPLANNING (ABSOLUTE PRIORITY):
   - The PRIMARY GOAL is your overriding priority across all runs and replanning cycles.
   - The execution plan is a flexible hypothesis, NOT a rigid contract.
   - You have full authority and are EXPECTED to change the plan, add subgoals, skip obsolete steps, or completely replan using 'update_plan' whenever necessary to accomplish the primary goal.
   - If an error occurs, dependencies are missing, or commands fail, DO WHATEVER IS NECESSARY: diagnose failures, rewrite code, test alternative workarounds, mutate the plan, and self-correct.
   - Never stop or declare completion until you have independently verified that the primary goal is completely accomplished.""" if goal_supremacy else """1. STRICT PLAN EXECUTION MODE (GOAL SUPREMACY DISABLED):
   - Goal supremacy is toggled OFF: you must strictly adhere to the defined execution plan.
   - Do NOT invoke 'update_plan' or mutate the step sequence. Follow the plan steps as defined."""

        return f"""You are YoYo, an autonomous agentic AI system operating inside a non-interactive batch OS environment.

CRITICAL OPERATIONAL CONSTRAINTS:
{planning_rule}

2. TECHNICAL SPECIFICATIONS & RESEARCH PAPERS:
   - When research papers, SRS documents, or PRDs are ingested, carefully adhere to their functional requirements, mathematical algorithms, and architectural constraints.
   - Use 'ingest_document' on demand to inspect sections, pseudocode, and parameters.

3. NON-INTERACTIVE BATCH ENVIRONMENT:
   - This process is executing on an automated batch queue (e.g. Slurm / PBS / Cloud Batch).
   - Standard input (stdin) is disconnected (/dev/null).
   - You CANNOT prompt the user for questions, confirmations, or approvals.
   - You MUST make optimal decisions independently to complete the goal end-to-end.

4. SANDBOX SECURITY ENFORCEMENT (Profile: {sandbox_profile.upper()}):
   - Workspace Root: {workspace_root}
   - Writes are strictly confined to the workspace and scratch directories.
   - Access to credentials, sensitive files (~/.ssh, .env), and dangerous commands (sudo, rm -rf /) is blocked.
   - Child processes have sensitive environment variables scrubbed.

5. AUTOMATED VERIFICATION:
   - For every modification, run automated verification (unit tests, syntax checks, or validation commands).
   - Verify all outcomes against the primary goal before concluding the job.
   - Output structured reasoning so the audit trail captures your complete Chain of Thought.

ASSOCIATED INPUT FILES:
{files_desc}
{docs_block}

AVAILABLE TOOLS:
{tools_desc}
"""

    @classmethod
    def format_step_prompt(
        cls,
        goal: str,
        step_number: int,
        plan_summary: str,
        last_action_result: str = "",
        errors: str = "",
    ) -> str:
        prompt = [
            f"PRIMARY GOAL (TOP PRIORITY): {goal}",
            f"CURRENT STEP: {step_number}",
            f"CURRENT PLAN STATUS:\n{plan_summary}",
            "GUIDELINE: The primary goal must be accomplished. You may change the plan and invoke 'update_plan' if needed.",
        ]
        if last_action_result:
            prompt.append(f"LAST ACTION RESULT:\n{last_action_result[:1000]}")
        if errors:
            prompt.append(f"DETECTED ERRORS / FAILURES:\n{errors[:1000]}")

        prompt.append(
            "\nDecide the next optimal action. Provide your Chain of Thought reasoning "
            "(Goal assessment & alignment, observation, trade-offs, and decision) followed by your tool invocation."
        )
        return "\n\n".join(prompt)

