"""
Unit tests for crash-resilient state checkpointing and mid-stream resumption.
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
)
from batch_agent.core.engine import BatchEngine
from batch_agent.core.state import BatchJobState
from batch_agent.intelligence.planner import ExecutionPlan, PlanStep, StepStatus


class TestCheckpointAndResume(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)

        # Create sample workspace file
        with open(os.path.join(self.ws, "data.txt"), "w") as f:
            f.write("sample batch data\n")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_checkpoint_serialization_and_deserialization(self):
        plan = ExecutionPlan(goal="Test Checkpoint Goal")
        step1 = plan.add_step("Step 1", "Inspect files", "inspect")
        step1.mark_completed("Done inspecting")
        step2 = plan.add_step("Step 2", "Run calculations", "execute")
        step2.mark_in_progress()

        state = BatchJobState(
            job_id="test-checkpoint-001",
            goal="Test Checkpoint Goal",
            workspace_root=self.ws,
            plan=plan,
            current_step_number=2,
            artifacts_produced=["artifact1.json"],
        )

        checkpoint_file = os.path.join(self.ws, ".batch_checkpoint.json")
        saved_path = state.save_checkpoint(checkpoint_file)
        self.assertTrue(os.path.exists(saved_path))

        loaded_state = BatchJobState.load_checkpoint(checkpoint_file)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.job_id, "test-checkpoint-001")
        self.assertEqual(loaded_state.current_step_number, 2)
        self.assertEqual(len(loaded_state.plan.steps), 2)
        self.assertEqual(loaded_state.plan.steps[0].status, StepStatus.COMPLETED)
        self.assertEqual(loaded_state.plan.steps[1].status, StepStatus.IN_PROGRESS)
        self.assertIn("artifact1.json", loaded_state.artifacts_produced)

    def test_engine_resume_from_checkpoint(self):
        # 1. Simulate an interrupted job by creating a checkpoint where Step 1 is completed
        plan = ExecutionPlan(goal="Two-stage batch task")
        s1 = plan.add_step("Inspect data", "Read data.txt", "inspect")
        s1.mark_completed("Inspected data.txt")
        s2 = plan.add_step("Generate summary report", "Write batch report", "report")

        checkpoint_path = os.path.join(self.ws, ".batch_checkpoint.json")
        state = BatchJobState(
            job_id="resumed-batch-job",
            goal="Two-stage batch task",
            workspace_root=self.ws,
            plan=plan,
            current_step_number=1,
        )
        state.save_checkpoint(checkpoint_path)

        # 2. Run engine with --resume
        audit_file = os.path.join(self.ws, "resume_audit.txt")
        exit_code = main([
            "--resume",
            "--workspace", self.ws,
            "--log-file", audit_file,
            "--checkpoint", checkpoint_path,
        ])

        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(audit_file))

        with open(audit_file, "r", encoding="utf-8") as f:
            audit_content = f.read()

        self.assertIn("Batch Job Resumed from Checkpoint", audit_content)
        self.assertIn("Resumed execution from step 1", audit_content)


if __name__ == "__main__":
    unittest.main()

