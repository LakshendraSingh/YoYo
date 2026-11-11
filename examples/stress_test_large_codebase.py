#!/usr/bin/env python3
"""
Stress-Test Benchmark: Tests YoYo against a ridiculously large synthetic codebase
containing thousands of nested modules, classes, and cross-package imports.

Usage:
    python3 examples/stress_test_large_codebase.py --files 1000
    python3 examples/stress_test_large_codebase.py --files 5000 --provider heuristic
"""

import argparse
import os
import sys
import tempfile
import time

# Ensure batch_agent is in python path
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


def generate_megarepo(workspace: str, num_files: int = 1000) -> str:
    """Generate a massive monorepo with nested modules and a buried needle function."""
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
                f"# Generated module {i}\n"
                f"class ServiceWorker_{i}:\n"
                f"    '''Worker instance {i} handling synthetic batch load.'''\n"
                f"    def __init__(self, multiplier: int = {i}):\n"
                f"        self.multiplier = multiplier\n\n"
                f"    def execute_task(self, value: int) -> int:\n"
                f"        return value * self.multiplier\n\n"
                f"def compute_hash_{i}(data: str) -> int:\n"
                f"    return len(data) + {i}\n"
            )

    # Place a hidden target function deep in the repository hierarchy
    target_dir = os.path.join(workspace, f"pkg_{num_files // 100}", "sub_25")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "analytics_engine.py")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(
            "def calculate_aggregate_metric(a: int, b: int) -> int:\n"
            "    '''Core algorithm hidden deep inside the massive monorepo.'''\n"
            "    return a * b\n"
        )

    # Create associated test suite in tests/
    tests_dir = os.path.join(workspace, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_path = os.path.join(tests_dir, "test_analytics.py")
    rel_pkg = f"pkg_{num_files // 100}.sub_25.analytics_engine"
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(
            f"import os\n"
            f"import sys\n"
            f"import unittest\n\n"
            f"# Ensure workspace root is in sys.path regardless of how python invokes the test\n"
            f"sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n\n"
            f"from {rel_pkg} import calculate_aggregate_metric\n\n"
            f"class TestAnalytics(unittest.TestCase):\n"
            f"    def test_metric(self):\n"
            f"        self.assertEqual(calculate_aggregate_metric(6, 7), 42)\n\n"
            f"if __name__ == '__main__':\n"
            f"    unittest.main()\n"
        )

    duration = time.time() - t0
    print(f"Generated {num_files:,} files across {num_files // 50 + 1} packages in {duration:.2f}s.")
    return "tests/test_analytics.py"


def run_benchmark(num_files: int, provider_name: str = "heuristic") -> int:
    tmp_dir = tempfile.TemporaryDirectory()
    ws = os.path.realpath(tmp_dir.name)
    txt_log = os.path.join(ws, "stress_audit.txt")
    csv_log = os.path.join(ws, "stress_audit.csv")

    try:
        test_file = generate_megarepo(ws, num_files=num_files)

        config = BatchAgentConfig(
            job_id=f"megarepo-stress-{num_files}",
            goal="Locate calculate_aggregate_metric in the codebase, verify AST symbols, run test_analytics.py, and produce audit report",
            associated_files=[test_file],
            workspace_root=ws,
            max_steps=10,
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
        print(f"STARTING YOYO SCALE BENCHMARK ON {num_files:,} REPOSITORY FILES")
        print("=" * 70)

        t_start = time.time()
        engine = BatchEngine(config)
        exit_code = engine.run()
        total_time = time.time() - t_start

        txt_size = os.path.getsize(txt_log) if os.path.exists(txt_log) else 0
        csv_size = os.path.getsize(csv_log) if os.path.exists(csv_log) else 0

        if os.path.exists(txt_log):
            print("\n--- AUDIT TRAIL TRACE EXCERPT ---")
            with open(txt_log, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
                # Show key steps
                for line in lines:
                    if any(k in line for k in ["=== STEP", "Phase:", "Decision:", "Tool:", "Exit Code:", "Result:", "SUCCESS", "FAILED"]):
                        print(f"  {line}")
            print("--- END AUDIT TRAIL TRACE ---\n")

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY:")
        print(f"  Codebase Size      : {num_files:,} files (~{num_files * 15:,} lines of code)")
        print(f"  Total Runtime      : {total_time:.2f} seconds")
        print(f"  Autonomous Steps   : {engine.state.current_step_number}")
        print(f"  Tokens Consumed    : {engine.state.cumulative_tokens_used}")
        print(f"  Job Exit Status    : {'SUCCESS (0)' if exit_code == 0 else f'FAILED ({exit_code})'}")
        print(f"  Audit Log (.txt)   : {txt_size:,} bytes")
        print(f"  Audit Log (.csv)   : {csv_size:,} bytes")
        print("=" * 70 + "\n")

        return exit_code

    finally:
        tmp_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description="YoYo Large Codebase Stress-Test Benchmark")
    parser.add_argument("--files", "-n", type=int, default=1000, help="Number of files to generate (e.g. 1000, 5000, 10000)")
    parser.add_argument("--provider", type=str, default="heuristic", choices=["heuristic", "gemini", "openai", "local_weights"])
    args = parser.parse_args()

    sys.exit(run_benchmark(num_files=args.files, provider_name=args.provider))


if __name__ == "__main__":
    main()

