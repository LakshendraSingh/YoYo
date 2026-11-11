"""
Unit and integration tests for Goal Priority, Dynamic Replanning,
and Adaptive Plan Mutation ('update_plan' tool and plan.txt synchronization).
"""

import os
import tempfile
import unittest

from batch_agent.cli import main
from batch_agent.config import (
    AuditConfig,
    BatchAgentConfig,
    LLMProviderType,
    ProcessConfig,
    SandboxConfig,
    SandboxProfile,
)
from batch_agent.core.engine import BatchEngine
from batch_agent.intelligence.dependency_planner import DependencyPlanner
from batch_agent.intelligence.planner import ExecutionPlan, StepStatus
from batch_agent.tools.base import ToolContext
from batch_agent.tools.plan_tool import UpdatePlanTool


class TestGoalPriorityAndReplanning(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.audit_log = os.path.join(self.ws, "audit.txt")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_update_plan_tool_add_and_insert_steps(self):
        plan = ExecutionPlan(goal="Process genomic dataset and generate charts")
        plan.add_step(title="Step 1: Ingest FASTQ", description="Ingest files", action_type="inspect")

        tool = UpdatePlanTool()
        context = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=None,  # type: ignore
            tracker=None,  # type: ignore
            runner=None,   # type: ignore
            plan=plan,
        )

        # 1. Add step
        res = tool.run({
            "action": "add_step",
            "reason": "Genomic normalization requires quality trimming before analysis",
            "title": "Quality Trimming",
            "description": "Trim low quality base pairs",
            "action_type": "execute",
        }, context)

        self.assertTrue(res.success)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[1].title, "Quality Trimming")
        self.assertEqual(len(plan.plan_mutations), 1)

        # 2. Insert step at position 0
        res2 = tool.run({
            "action": "insert_step",
            "reason": "Emergency check for disk space before processing large files",
            "title": "Verify Disk Space",
            "description": "Check free inodes and gigabytes",
            "action_type": "verify",
            "insert_at": 0,
        }, context)

        self.assertTrue(res2.success)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].title, "Verify Disk Space")

    def test_update_plan_tool_skip_and_replace_remaining(self):
        plan = ExecutionPlan(goal="Build machine learning pipeline")
        step1 = plan.add_step(title="Data prep", description="", action_type="execute")
        step1.mark_completed("Data ready")
        plan.current_step_index = 1
        plan.add_step(title="Obsolete preprocessing", description="", action_type="execute")
        plan.add_step(title="Train model", description="", action_type="execute")

        tool = UpdatePlanTool()
        context = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=None,  # type: ignore
            tracker=None,  # type: ignore
            runner=None,   # type: ignore
            plan=plan,
        )

        # Skip obsolete step
        res_skip = tool.run({
            "action": "skip_step",
            "step_id": 2,
            "reason": "Preprocessing already performed upstream in data prep",
        }, context)
        self.assertTrue(res_skip.success)
        self.assertEqual(plan.steps[1].status, StepStatus.SKIPPED)

        # Replace remaining steps
        res_replace = tool.run({
            "action": "replace_remaining",
            "reason": "Switching from standard regression to gradient boosting for superior goal accuracy",
            "new_steps": [
                {"title": "Train XGBoost", "description": "Train boosted trees", "action_type": "execute"},
                {"title": "Evaluate AUC", "description": "Compute ROC-AUC score", "action_type": "verify"},
            ],
        }, context)
        self.assertTrue(res_replace.success)
        # Should keep completed step1, and replace remaining with 2 new steps
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].title, "Data prep")
        self.assertEqual(plan.steps[1].title, "Train XGBoost")
        self.assertEqual(plan.steps[2].title, "Evaluate AUC")

    def test_update_plan_syncs_to_disk_plan_txt(self):
        # Create an initial plan.txt on disk
        plan_path = os.path.join(self.ws, "plan.txt")
        initial_plan = (
            "================================================================================\n"
            "YOYO - OFFLINE EXECUTION PLAN & DEPENDENCY MANIFEST\n"
            "================================================================================\n"
            "Primary Goal        : Original Goal\n"
            "\n"
            "[SECTION 6: EXECUTION SUBGOALS]\n"
            "  1. [PENDING] Baseline check\n"
            "================================================================================\n"
        )
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(initial_plan)

        plan = ExecutionPlan(goal="Updated Priority Goal")
        step1 = plan.add_step(title="Baseline check", description="", action_type="verify")
        step1.mark_completed("Pass")

        tool = UpdatePlanTool()
        context = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=None,  # type: ignore
            tracker=None,  # type: ignore
            runner=None,   # type: ignore
            plan=plan,
        )

        # Agent mutates plan to achieve goal
        res = tool.run({
            "action": "add_step",
            "reason": "Crucial post-processing required to satisfy primary goal",
            "title": "Post-process Results",
            "description": "Clean and format tabular output",
            "action_type": "execute",
        }, context)

        self.assertTrue(res.success)
        # Check plan.txt on disk was updated
        with open(plan_path, "r", encoding="utf-8") as f:
            disk_content = f.read()

        self.assertIn("Primary Goal        : Updated Priority Goal", disk_content)
        self.assertIn("[COMPLETED] Baseline check", disk_content)
        self.assertIn("[PENDING] Post-process Results", disk_content)
        self.assertIn("[SECTION 7: GOAL PRIORITY & ADAPTIVE PLAN MUTATION HISTORY]", disk_content)
        self.assertIn("Crucial post-processing required to satisfy primary goal", disk_content)

    def test_cli_allows_goal_with_plan_override(self):
        # Create plan.txt with old goal
        plan_file = os.path.join(self.ws, "plan.txt")
        with open(plan_file, "w", encoding="utf-8") as f:
            f.write(
                "Primary Goal        : Old Goal From Previous Run\n"
                "[SECTION 6: EXECUTION SUBGOALS]\n"
                "  1. [PENDING] Ingest telemetry\n"
                "  2. [PENDING] Run verification\n"
                "================================================================================\n"
            )

        # Run CLI providing BOTH --plan and a new --goal
        # The new --goal MUST take priority!
        log_file = os.path.join(self.ws, "cli_goal_priority_audit.txt")
        exit_code = main([
            "--plan", plan_file,
            "--goal", "Overriding Priority Goal: Compute 99th Percentile Latency",
            "--workspace", self.ws,
            "--log-file", log_file,
            "--max-steps", "5",
        ])

        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r", encoding="utf-8") as f:
            audit_text = f.read()

        self.assertIn("Overriding Priority Goal: Compute 99th Percentile Latency", audit_text)

    def test_engine_dynamic_goal_verification(self):
        # Verify that if a proposal test suite exists, the engine ensures verification is executed
        proposal_dir = os.path.join(self.ws, "proposal")
        os.makedirs(proposal_dir, exist_ok=True)
        with open(os.path.join(proposal_dir, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nclass T(unittest.TestCase):\n    def test_1(self): self.assertTrue(True)\n")

        config = BatchAgentConfig(
            job_id="goal-priority-job-999",
            goal="Ensure full system verification and produce summary",
            workspace_root=self.ws,
            max_steps=6,
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(enabled=True, profile=SandboxProfile.WORKSPACE, workspace_root=self.ws),
            process=ProcessConfig(default_timeout_seconds=10),
            audit=AuditConfig(log_file_path=self.audit_log, flush_immediately=True),
        )

        engine = BatchEngine(config)
        code = engine.run()
        self.assertEqual(code, 0)

        with open(self.audit_log, "r", encoding="utf-8") as f:
            log_content = f.read()

        self.assertIn("Primary Goal Priority", log_content)
        self.assertIn("SUCCESS", log_content)


if __name__ == "__main__":
    unittest.main()
