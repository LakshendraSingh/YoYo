"""
Unit tests validating scale, memory resilience, and search performance on large codebases.
"""

import os
import sys
import tempfile
import time
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


class TestLargeCodebaseScale(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.txt_log = os.path.join(self.ws, "scale_audit.txt")
        self.csv_log = os.path.join(self.ws, "scale_audit.csv")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_navigation_and_execution_on_large_codebase(self):
        # Generate 500 modules across nested directory structures
        num_files = 500
        for i in range(num_files):
            d = os.path.join(self.ws, f"package_{i // 25}", f"sub_{i % 25}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"worker_{i}.py"), "w", encoding="utf-8") as f:
                f.write(f"def process_{i}(n: int) -> int:\n    return n * {i}\n")

        # Plant target module and test
        target_dir = os.path.join(self.ws, "package_10", "sub_5")
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "metric_calc.py"), "w", encoding="utf-8") as f:
            f.write("def calculate_metric(a: int, b: int) -> int:\n    return a * b\n")

        tests_dir = os.path.join(self.ws, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        test_file = os.path.join(tests_dir, "test_metric.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(
                "import unittest\n"
                "from package_10.sub_5.metric_calc import calculate_metric\n\n"
                "class TestM(unittest.TestCase):\n"
                "    def test_calc(self):\n"
                "        self.assertEqual(calculate_metric(4, 5), 20)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )

        config = BatchAgentConfig(
            job_id="test-scale-500",
            goal="Find calculate_metric, inspect AST, run verification, and record audit trail",
            associated_files=["tests/test_metric.py"],
            workspace_root=self.ws,
            max_steps=8,
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(enabled=True, profile=SandboxProfile.WORKSPACE, workspace_root=self.ws),
            process=ProcessConfig(default_timeout_seconds=20),
            audit=AuditConfig(log_file_path=self.txt_log, csv_log_path=self.csv_log, enable_csv_log=True),
        )

        t_start = time.time()
        engine = BatchEngine(config)
        exit_code = engine.run()
        elapsed = time.time() - t_start

        self.assertEqual(exit_code, 0)
        self.assertTrue(engine.state.is_success)
        self.assertLess(elapsed, 10.0, f"Scale test should finish fast, took {elapsed:.2f}s")
        self.assertTrue(os.path.exists(self.txt_log))
        self.assertTrue(os.path.exists(self.csv_log))
        self.assertGreater(os.path.getsize(self.txt_log), 0)
        self.assertGreater(os.path.getsize(self.csv_log), 0)


if __name__ == "__main__":
    unittest.main()

