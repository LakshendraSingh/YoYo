"""
Unit tests for .txt audit trail logging, formatting, and disk flush verification.
"""

import os
import tempfile
import unittest

from batch_agent.audit.audit_logger import AuditLogger
from batch_agent.audit.events import ProcessSnapshot
from batch_agent.config import AuditConfig


class TestAuditLogger(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.log_file = os.path.join(self.ws, "test_audit.txt")
        self.config = AuditConfig(log_file_path=self.log_file, flush_immediately=True)
        self.logger = AuditLogger(self.config, self.ws, job_id="job-test-123")

    def tearDown(self):
        self.logger.close()
        self.tmp_dir.cleanup()

    def test_log_file_created_and_written(self):
        self.logger.log_job_start(
            goal="Test Goal for Batch OS",
            associated_files=["file1.py", "data.csv"],
            sandbox_profile="workspace",
            max_steps=10,
            provider="heuristic",
        )

        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("YOYO - AUDIT TRAIL [JOB: job-test-123]", content)
        self.assertIn("Goal Specification : Test Goal for Batch OS", content)
        self.assertIn("NON-INTERACTIVE BATCH OS ENVIRONMENT", content)

    def test_chain_of_thought_logged_with_structure(self):
        self.logger.log_chain_of_thought(
            step=1,
            phase="PLANNING",
            reasoning="Goal Assessment: Ingest telemetry.\nObservation: 5 records.\nDecision: Parse rows.",
            plan_summary="[ ] Step 1: Parse rows",
            decision="Execute parsing step",
        )

        with open(self.log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("[CHAIN_OF_THOUGHT] [STEP 1]", content)
        self.assertIn("[REASONING / CHAIN OF THOUGHT]", content)
        self.assertIn("Goal Assessment: Ingest telemetry", content)
        self.assertIn("Optimal Decision Taken:", content)
        self.assertIn("Execute parsing step", content)

    def test_command_execution_and_active_processes_logged(self):
        snapshot = ProcessSnapshot(
            pid=9999,
            command="python3 -m unittest",
            status="RUNNING",
            start_time="2026-11-11T00:00:00Z",
            duration_seconds=1.5,
        )

        self.logger.log_command_execution(
            step=2,
            command="python3 -m unittest tests/test_calc.py",
            working_dir=self.ws,
            exit_code=0,
            stdout="Ran 5 tests in 0.12s\nOK",
            stderr="",
            duration_seconds=0.12,
            active_processes=[snapshot],
            sanitized_env_keys=["GEMINI_API_KEY", "OPENAI_API_KEY"],
        )

        with open(self.log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("[COMMAND_EXECUTION] [STEP 2]", content)
        self.assertIn("Command     : python3 -m unittest tests/test_calc.py", content)
        self.assertIn("Exit Code   : 0", content)
        self.assertIn("Active Subprocesses Snapshot:", content)
        self.assertIn("9999", content)
        self.assertIn("python3 -m unittest", content)
        self.assertIn("sanitized_env_keys_stripped:", content)

    def test_job_completion_logged(self):
        self.logger.log_job_completion(
            success=True,
            total_steps=5,
            total_runtime_seconds=12.34,
            final_summary="All subgoals completed without errors.",
            artifacts_produced=["report.md", "data.json"],
            active_processes=[],
        )

        with open(self.log_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("AUDIT TRAIL COMPLETE - FINAL SUMMARY", content)
        self.assertIn("Final Job Status   : SUCCESS", content)
        self.assertIn("Total Steps Taken  : 5", content)
        self.assertIn("Produced Artifacts : report.md, data.json", content)


if __name__ == "__main__":
    unittest.main()
