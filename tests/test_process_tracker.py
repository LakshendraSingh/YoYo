"""
Unit tests for process tracking, non-interactive execution, and timeouts.
"""

import os
import subprocess
import tempfile
import time
import unittest

from batch_agent.config import ProcessConfig, SandboxConfig
from batch_agent.process.runner import ProcessRunner
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager


class TestProcessTracker(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.sandbox = SandboxManager(SandboxConfig(workspace_root=self.ws), ProcessConfig())
        self.tracker = ProcessTracker()
        self.runner = ProcessRunner(
            sandbox=self.sandbox,
            tracker=self.tracker,
            config=ProcessConfig(default_timeout_seconds=5),
        )

    def tearDown(self):
        self.tracker.terminate_all()
        self.tmp_dir.cleanup()

    def test_run_command_success(self):
        result = self.runner.run("echo 'YoYo Running'", cwd=self.ws)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("YoYo Running", result.stdout)
        self.assertFalse(result.timed_out)

    def test_active_process_registration_and_cleanup(self):
        # Verify tracker registers and completes
        result = self.runner.run("python3 -c 'import time; time.sleep(0.1); print(\"done\")'")
        self.assertEqual(result.exit_code, 0)
        records = self.tracker.get_all_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "COMPLETED")
        self.assertEqual(len(self.tracker.get_active_snapshots()), 0)

    def test_timeout_enforcement(self):
        # Run command exceeding short timeout
        result = self.runner.run(
            "python3 -c 'import time; time.sleep(10)'",
            timeout_seconds=1,
        )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)
        self.assertIn("timed out after 1 seconds", result.stderr)
        self.assertEqual(len(self.tracker.get_active_snapshots()), 0)

    def test_non_interactive_stdin_never_hangs(self):
        # Command that attempts to read from stdin: in non-interactive batch environment with DEVNULL,
        # it should terminate immediately with EOFError instead of hanging!
        result = self.runner.run(
            "python3 -c 'try:\n  s = input(\"prompt:\")\nexcept EOFError:\n  print(\"EOF received successfully\")'",
            timeout_seconds=3,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("EOF received successfully", result.stdout)


if __name__ == "__main__":
    unittest.main()

