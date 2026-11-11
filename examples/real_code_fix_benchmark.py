#!/usr/bin/env python3
"""
Real Autonomous Bug-Fix Benchmark on Large Codebase (1,000 Files).
Tests if Gemini can genuinely diagnose a failing test, locate the broken source file,
use file_edit to repair the bug, re-test, and verify success without human intervention.
"""

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from batch_agent.config import (
    AuditConfig,
    BatchAgentConfig,
    LLMProviderType,
    ProcessConfig,
    SandboxConfig,
    SandboxProfile,
)
from batch_agent.core.engine import BatchEngine


def generate_buggy_megarepo(workspace: str, num_files: int = 1000) -> str:
    """Generates 1,000 repository files with a planted algorithmic bug in the target module."""
    print(f"Generating synthetic codebase with {num_files:,} Python files...")
    t0 = time.time()

    for i in range(num_files):
        pkg_id = i // 50
        sub_id = i % 50
        dir_path = os.path.join(workspace, f"pkg_{pkg_id}", f"sub_{sub_id}")
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"module_{i}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Generated worker module {i}\n"
                f"class ServiceWorker_{i}:\n"
                f"    def __init__(self, multiplier: int = {i}):\n"
                f"        self.multiplier = multiplier\n"
                f"    def execute_task(self, value: int) -> int:\n"
                f"        return value * self.multiplier\n"
            )

    # Plant defective function deep in the repository hierarchy
    target_dir = os.path.join(workspace, f"pkg_{num_files // 100}", "sub_25")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "analytics_engine.py")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(
            "def calculate_aggregate_metric(a: int, b: int) -> int:\n"
            "    \"\"\"Computes the aggregate product of a and b.\"\"\"\n"
            "    return a + b  # DEFECT: Addition instead of multiplication\n"
        )

    # Create associated test suite expecting multiplication (6 * 7 == 42)
    tests_dir = os.path.join(workspace, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_path = os.path.join(tests_dir, "test_analytics.py")
    rel_pkg = f"pkg_{num_files // 100}.sub_25.analytics_engine"
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(
            f"import os\n"
            f"import sys\n"
            f"import unittest\n\n"
            f"sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n\n"
            f"from {rel_pkg} import calculate_aggregate_metric\n\n"
            f"class TestAnalytics(unittest.TestCase):\n"
            f"    def test_metric(self):\n"
            f"        # 6 * 7 must equal 42, but defective code returns 6 + 7 = 13\n"
            f"        self.assertEqual(calculate_aggregate_metric(6, 7), 42)\n\n"
            f"if __name__ == '__main__':\n"
            f"    unittest.main()\n"
        )

    duration = time.time() - t0
    print(f"Generated {num_files:,} files in {duration:.2f}s.")
    print("PLANTED DEFECT: calculate_aggregate_metric returns a + b (13) instead of a * b (42).")
    return "tests/test_analytics.py"


def run_bug_fix_benchmark(num_files: int, provider_name: str = "gemini") -> int:
    tmp_dir = tempfile.TemporaryDirectory()
    ws = os.path.realpath(tmp_dir.name)
    txt_log = os.path.join(ws, "bugfix_audit.txt")
    csv_log = os.path.join(ws, "bugfix_audit.csv")

    try:
        test_file = generate_buggy_megarepo(ws, num_files=num_files)

        config = BatchAgentConfig(
            job_id=f"real-bugfix-{num_files}",
            goal=(
                "The test suite tests/test_analytics.py is currently FAILING. "
                "Investigate the failure, locate the defective source file in the codebase, "
                "use file_edit to repair the bug, re-run verification tests to prove it passes, "
                "and complete the job."
            ),
            associated_files=[test_file],
            workspace_root=ws,
            max_steps=12,
            provider=LLMProviderType(provider_name),
            api_key=os.environ.get("GEMINI_API_KEY"),
            sandbox=SandboxConfig(
                enabled=True,
                profile=SandboxProfile.WORKSPACE,
                workspace_root=ws,
            ),
            process=ProcessConfig(default_timeout_seconds=30),
            audit=AuditConfig(
                log_file_path=txt_log,
                csv_log_path=csv_log,
                enable_csv_log=True,
            ),
        )

        print("\n" + "=" * 70)
        print(f"STARTING REAL AUTONOMOUS BUG FIX BENCHMARK ON {num_files:,} FILES")
        print("=" * 70)

        t_start = time.time()
        engine = BatchEngine(config)
        exit_code = engine.run()
        total_time = time.time() - t_start

        # Check target file on disk to see if Gemini actually modified the code
        target_path = os.path.join(ws, f"pkg_{num_files // 100}", "sub_25", "analytics_engine.py")
        fixed_content = ""
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                fixed_content = f.read()

        txt_size = os.path.getsize(txt_log) if os.path.exists(txt_log) else 0

        print("\n" + "=" * 70)
        print("REAL BENCHMARK RESULTS SUMMARY:")
        print(f"  Codebase Size        : {num_files:,} files (~15,000 lines of code)")
        print(f"  Total Runtime        : {total_time:.2f} seconds")
        print(f"  Autonomous Steps     : {engine.state.current_step_number}")
        print(f"  Tokens Consumed      : {engine.state.cumulative_tokens_used}")
        print(f"  Job Exit Status      : {'SUCCESS (0)' if exit_code == 0 else f'FAILED ({exit_code})'}")
        print(f"  Repaired Code File   : \n{fixed_content}")
        print(f"  Audit Log Size       : {txt_size:,} bytes")
        print("=" * 70 + "\n")

        return exit_code

    finally:
        tmp_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description="YoYo Real Bug-Fix Benchmark")
    parser.add_argument("--files", "-n", type=int, default=1000, help="Number of files to generate")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "heuristic", "openai"])
    args = parser.parse_args()

    sys.exit(run_bug_fix_benchmark(num_files=args.files, provider_name=args.provider))


if __name__ == "__main__":
    main()

