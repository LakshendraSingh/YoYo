"""
End-to-end integration tests for autonomous, non-interactive batch job execution.
Verifies goal completion without human intervention and full .txt audit trail capture.
"""

import os
import shutil
import tempfile
import unittest

from batch_agent.config import (
    AuditConfig,
    BatchAgentConfig,
    LLMProviderType,
    ProcessConfig,
    SandboxConfig,
    SandboxProfile,
)
from batch_agent.core.engine import BatchEngine


class TestAutonomousEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)

        # Create sample files inside workspace
        self.data_dir = os.path.join(self.ws, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.sample_csv = os.path.join(self.data_dir, "samples.csv")
        with open(self.sample_csv, "w") as f:
            f.write("id,score\n1,100\n2,85\n3,92\n")

        # Create a sample python test file
        self.tests_dir = os.path.join(self.ws, "tests")
        os.makedirs(self.tests_dir, exist_ok=True)
        with open(os.path.join(self.tests_dir, "test_sample.py"), "w") as f:
            f.write("import unittest\n\nclass TestSample(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(1+1, 2)\n")

        self.audit_log_path = os.path.join(self.ws, "audit_trail.txt")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_end_to_end_autonomous_batch_execution(self):
        config = BatchAgentConfig(
            job_id="test-e2e-job-001",
            goal="Analyze sample data, run verification tests, and produce batch summary",
            associated_files=["data/samples.csv"],
            workspace_root=self.ws,
            max_steps=10,
            job_timeout_seconds=60,
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(
                enabled=True,
                profile=SandboxProfile.WORKSPACE,
                workspace_root=self.ws,
            ),
            process=ProcessConfig(default_timeout_seconds=10),
            audit=AuditConfig(
                log_file_path=self.audit_log_path,
                flush_immediately=True,
                record_active_processes_snapshot=True,
            ),
        )

        engine = BatchEngine(config)
        exit_code = engine.run()

        # 1. Check exit code is 0 (Success)
        self.assertEqual(exit_code, 0, f"Engine should exit with 0, got {exit_code}")
        self.assertTrue(engine.state.is_success)

        # 2. Verify audit log exists
        self.assertTrue(os.path.exists(self.audit_log_path), "Audit log file must be created")

        with open(self.audit_log_path, "r", encoding="utf-8") as f:
            audit_text = f.read()

        # 3. Verify Chain of Thought captured
        self.assertIn("[CHAIN_OF_THOUGHT]", audit_text)
        self.assertIn("Goal Assessment", audit_text)
        self.assertIn("Optimal Decision Taken", audit_text)

        # 4. Verify Active Subprocesses Snapshot captured
        self.assertIn("Active Subprocesses Snapshot:", audit_text)

        # 5. Verify Tool Invocations captured
        self.assertIn("[TOOL_INVOCATION]", audit_text)

        # 6. Verify Final Summary Banner
        self.assertIn("AUDIT TRAIL COMPLETE - FINAL SUMMARY", audit_text)
        self.assertIn("Final Job Status   : SUCCESS", audit_text)

        # 7. Verify artifacts created
        summary_artifact = os.path.join(self.ws, "batch_summary.md")
        self.assertTrue(os.path.exists(summary_artifact), "batch_summary.md artifact should be generated")


if __name__ == "__main__":
    unittest.main()

