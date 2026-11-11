"""
Comprehensive tests for:
1. Runaway Loop Risk Defenses (hard step limits, token budgets, infinite retry circuit breakers).
2. Dual Audit Logging (.txt and .csv with strict timestamps and machine-parseable delimiters).
3. Autonomous Command Failure Detection and Self-Healing Recovery.
4. Goal Supremacy Toggle (enabled vs strict-plan mode).
"""

import csv
import os
import sys
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
from batch_agent.intelligence.planner import ExecutionPlan
from batch_agent.process.recovery import (
    CommandRecoveryEngine,
    FailureCategory,
    ProcessResult,
)
from batch_agent.process.runner import ProcessRunner
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager
from batch_agent.tools.base import ToolContext
from batch_agent.tools.execution import CommandExecutionTool
from batch_agent.tools.plan_tool import UpdatePlanTool


class TestRunawayLoopAndRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.log_txt = os.path.join(self.ws, "audit.txt")
        self.log_csv = os.path.join(self.ws, "audit.csv")

    def tearDown(self):
        self.tmp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Dual Audit Logging (.txt and .csv)
    # -------------------------------------------------------------------------
    def test_dual_audit_logging_produces_valid_csv_and_txt(self):
        config = BatchAgentConfig(
            job_id="test-dual-log-01",
            goal="Test dual logging mechanism",
            workspace_root=self.ws,
            max_steps=6,
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(enabled=True, profile=SandboxProfile.WORKSPACE, workspace_root=self.ws),
            process=ProcessConfig(default_timeout_seconds=5),
            audit=AuditConfig(log_file_path=self.log_txt, csv_log_path=self.log_csv),
        )

        engine = BatchEngine(config)
        code = engine.run()
        self.assertEqual(code, 0)

        # Verify .txt log exists and is structured
        self.assertTrue(os.path.exists(self.log_txt))
        with open(self.log_txt, "r", encoding="utf-8") as f:
            txt_content = f.read()
        self.assertIn("YOYO - AUDIT TRAIL", txt_content)

        # Verify .csv log exists and is strictly parseable
        self.assertTrue(os.path.exists(self.log_csv))
        with open(self.log_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 0)
        first = rows[0]
        self.assertIn("timestamp", first)
        self.assertIn("event_type", first)
        self.assertIn("status", first)
        self.assertIn("duration_seconds", first)
        self.assertIn("tokens_used", first)

        # Verify all timestamps are non-empty ISO strings
        for row in rows:
            self.assertTrue(len(row["timestamp"]) > 10)
            self.assertIn(row["status"], ("SUCCESS", "FAILED", "TIMED_OUT", "RECOVERED", "INTERCEPTED", ""))

    # -------------------------------------------------------------------------
    # 2. Runaway Loop Defenses: Token Budget Limit
    # -------------------------------------------------------------------------
    def test_runaway_loop_token_budget_circuit_breaker(self):
        # Set an intentionally tiny token budget that trips on first or second step
        config = BatchAgentConfig(
            job_id="token-guard-job",
            goal="Simulate runaway model loop to test token budget tripping",
            workspace_root=self.ws,
            max_steps=10,
            max_token_budget=150,  # Trips rapidly
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(enabled=True, profile=SandboxProfile.WORKSPACE, workspace_root=self.ws),
            process=ProcessConfig(default_timeout_seconds=5),
            audit=AuditConfig(log_file_path=self.log_txt, csv_log_path=self.log_csv),
        )

        engine = BatchEngine(config)
        code = engine.run()

        # Engine should trip circuit breaker and exit with 1
        self.assertEqual(code, 1)
        self.assertIn("Token budget limit reached", engine.state.error_message or "")

        # Verify circuit breaker event recorded in CSV log
        with open(self.log_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            event_types = [r["event_type"] for r in reader]
        self.assertIn("CIRCUIT_BREAKER", event_types)

    # -------------------------------------------------------------------------
    # 3. Runaway Loop Defenses: Infinite Retry Circuit Breaker
    # -------------------------------------------------------------------------
    def test_infinite_retry_circuit_breaker_trips_on_consecutive_failures(self):
        # Create a failing command step and simulate consecutive identical failures
        plan = ExecutionPlan(goal="Test loop breaker")
        plan.add_step(title="Bad Step", description="", action_type="execute")

        config = BatchAgentConfig(
            job_id="retry-breaker-job",
            goal="Infinite loop breaker test",
            workspace_root=self.ws,
            max_steps=10,
            max_consecutive_failed_actions=2,  # Trips after 2 failures
            goal_supremacy=False,              # In strict mode, trips and stops
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(enabled=True, profile=SandboxProfile.WORKSPACE, workspace_root=self.ws),
            process=ProcessConfig(default_timeout_seconds=5),
            audit=AuditConfig(log_file_path=self.log_txt, csv_log_path=self.log_csv),
        )

        engine = BatchEngine(config)
        engine.plan = plan

        # Mock provider that repeatedly attempts the exact same failing command
        class StuckProvider:
            def decide_next_step(self, **kwargs):
                from batch_agent.intelligence.cot_engine import ChainOfThoughtEngine
                from batch_agent.intelligence.provider import AgentDecision
                return AgentDecision(
                    thought=ChainOfThoughtEngine.create_thought(
                        phase="ACTING", goal_analysis="stuck", observation="stuck", decision="stuck"
                    ),
                    tool_name="run_command",
                    tool_params={"command": "false"},  # always fails
                )

        engine.provider = StuckProvider()
        code = engine.run()

        self.assertEqual(code, 1)
        self.assertIn("failed 2 consecutive times", engine.state.error_message or "")

    # -------------------------------------------------------------------------
    # 4. Autonomous Command Failure Diagnosis & Recovery
    # -------------------------------------------------------------------------
    def test_command_recovery_engine_diagnostics(self):
        # 1. Test Exit 127 Command Not Found
        res_127 = ProcessResult(command="mytool --flag", exit_code=127, stdout="", stderr="zsh: command not found: mytool", duration_seconds=0.01)
        cat, diag = CommandRecoveryEngine.diagnose_failure("mytool --flag", res_127)
        self.assertEqual(cat, FailureCategory.COMMAND_NOT_FOUND)
        self.assertIn("mytool", diag)

        # 2. Test Missing Dependency
        res_dep = ProcessResult(command="python3 compute.py", exit_code=1, stdout="", stderr="ModuleNotFoundError: No module named 'scipy'", duration_seconds=0.05)
        cat_dep, diag_dep = CommandRecoveryEngine.diagnose_failure("python3 compute.py", res_dep)
        self.assertEqual(cat_dep, FailureCategory.MISSING_DEPENDENCY)
        self.assertIn("scipy", diag_dep)

        # 3. Test File Not Found
        res_fnf = ProcessResult(command="python3 parse.py data/raw.csv", exit_code=1, stdout="", stderr="FileNotFoundError: [Errno 2] No such file or directory: 'data/raw.csv'", duration_seconds=0.02)
        cat_fnf, diag_fnf = CommandRecoveryEngine.diagnose_failure("python3 parse.py data/raw.csv", res_fnf)
        self.assertEqual(cat_fnf, FailureCategory.FILE_OR_PATH_NOT_FOUND)
        self.assertIn("data/raw.csv", diag_fnf)

        # 4. Test Timeout
        res_timeout = ProcessResult(command="python3 train.py", exit_code=124, stdout="", stderr="", timed_out=True, duration_seconds=60.0)
        cat_to, diag_to = CommandRecoveryEngine.diagnose_failure("python3 train.py", res_timeout)
        self.assertEqual(cat_to, FailureCategory.TIMEOUT_EXPIRED)

    def test_autonomous_recovery_from_shifted_file_path(self):
        """Test agent recovers when command specifies wrong path but file exists elsewhere in workspace."""
        # Create a file in a subfolder: 'src/calc.py'
        src_dir = os.path.join(self.ws, "src")
        os.makedirs(src_dir, exist_ok=True)
        calc_file = os.path.join(src_dir, "calc.py")
        with open(calc_file, "w") as f:
            f.write("print('Autonomous recovery path success!')\n")

        sandbox = SandboxManager(SandboxConfig(workspace_root=self.ws), ProcessConfig())
        tracker = ProcessTracker()
        runner = ProcessRunner(sandbox=sandbox, tracker=tracker, config=ProcessConfig())

        context = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=sandbox,
            tracker=tracker,
            runner=runner,
        )

        tool = CommandExecutionTool()
        # Erroneous command references 'calc.py' directly at root (does not exist at root)
        wrong_cmd = f"{sys.executable} calc.py"
        res = tool.run({"command": wrong_cmd}, context)

        # The autonomous recovery engine should find 'src/calc.py', rewrite the command, and succeed!
        self.assertTrue(res.success, f"Tool should have autonomously recovered: {res.output}")
        self.assertIn("[AUTONOMOUS RECOVERY SUCCESSFUL]", res.output)
        self.assertIn("Autonomous recovery path success!", res.output)

    def test_pytest_translated_to_unittest_without_fake_stub(self):
        # Verify that running a missing pytest command does NOT create a dummy pytest.py stub,
        # but instead translates to python -m unittest and actually executes the test!
        tracker = ProcessTracker()
        sandbox = SandboxManager(SandboxConfig(enabled=False))
        runner = ProcessRunner(sandbox=sandbox, tracker=tracker, config=ProcessConfig(default_timeout_seconds=10))
        context = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=sandbox,
            tracker=tracker,
            runner=runner,
        )

        test_dir = os.path.join(self.ws, "tests")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_sample.py")
        with open(test_file, "w") as f:
            f.write("import unittest\nclass T(unittest.TestCase):\n    def test_pass(self): self.assertEqual(2+2, 4)\n")

        tool = CommandExecutionTool()
        cmd = f"python3 -m pytest tests/test_sample.py"
        res = tool.run({"command": cmd}, context)

        # 1. Must NOT create a dummy pytest.py stub
        self.assertFalse(os.path.exists(os.path.join(self.ws, "pytest.py")), "Never create dummy stub for pytest!")

        # 2. Must succeed by translating to unittest
        self.assertTrue(res.success, f"Should recover by translating to unittest: {res.output}")
        self.assertIn("Translated command to Python standard library unittest", res.output)

    # -------------------------------------------------------------------------
    # 5. Goal Supremacy Toggle (Enabled vs Strict Plan Mode)
    # -------------------------------------------------------------------------
    def test_goal_supremacy_toggle_enforces_strict_plan_mode(self):
        plan = ExecutionPlan(goal="Strict Plan Goal")
        plan.add_step(title="Fixed Step 1", description="", action_type="execute")

        tool = UpdatePlanTool()

        # Case A: Strict Plan Mode (goal_supremacy = False)
        class MockEngineStrict:
            class Config:
                goal_supremacy = False
            config = Config()

        context_strict = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=None,  # type: ignore
            tracker=None,  # type: ignore
            runner=None,   # type: ignore
            plan=plan,
            engine=MockEngineStrict(),
        )

        res_strict = tool.run({"action": "add_step", "reason": "Attempt mutation", "title": "New Step"}, context_strict)
        self.assertFalse(res_strict.success)
        self.assertIn("Goal Supremacy toggle is disabled", res_strict.error or "")
        self.assertEqual(len(plan.steps), 1)

        # Case B: Goal Supremacy Mode (goal_supremacy = True)
        class MockEngineSupremacy:
            class Config:
                goal_supremacy = True
            config = Config()

        context_supremacy = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=None,  # type: ignore
            tracker=None,  # type: ignore
            runner=None,   # type: ignore
            plan=plan,
            engine=MockEngineSupremacy(),
        )

        res_supremacy = tool.run({"action": "add_step", "reason": "Legitimate mutation", "title": "New Step"}, context_supremacy)
        self.assertTrue(res_supremacy.success)
        self.assertEqual(len(plan.steps), 2)

    # -------------------------------------------------------------------------
    # 6. Runaway Loop Risk Defenses Toggles (CLI & Config)
    # -------------------------------------------------------------------------
    def test_runaway_loop_risk_defenses_toggleable(self):
        # Verify CLI flags correctly toggle all defenses
        from batch_agent.cli import create_parser
        parser = create_parser()

        args = parser.parse_args([
            "--goal", "Test toggles",
            "--no-step-limit",
            "--no-timeout",
            "--no-token-budget",
            "--no-subprocess-timeout",
            "--no-circuit-breaker",
            "--no-recovery",
            "--strict-plan",
        ])

        self.assertTrue(args.no_step_limit)
        self.assertTrue(args.no_timeout)
        self.assertTrue(args.no_token_budget)
        self.assertTrue(args.no_subprocess_timeout)
        self.assertTrue(args.no_circuit_breaker)
        self.assertFalse(args.enable_recovery)
        self.assertFalse(args.goal_supremacy)

        # Test engine runs with disabled defenses without crashing or premature tripping
        config = BatchAgentConfig(
            job_id="test-toggles-disabled",
            goal="Simple verification",
            workspace_root=self.ws,
            max_steps=0,               # Disabled step limit
            job_timeout_seconds=0,     # Disabled wall-clock timeout
            max_token_budget=0,        # Disabled token budget
            max_consecutive_failed_actions=0,  # Disabled circuit breaker
            enable_recovery_engine=False,
            provider=LLMProviderType.HEURISTIC,
            sandbox=SandboxConfig(enabled=True, profile=SandboxProfile.WORKSPACE, workspace_root=self.ws),
            process=ProcessConfig(default_timeout_seconds=0),
            audit=AuditConfig(log_file_path=self.log_txt, enable_csv_log=False),
        )
        engine = BatchEngine(config)
        exit_code = engine.run()
        self.assertEqual(exit_code, 0)
        self.assertTrue(engine.state.is_success)


if __name__ == "__main__":
    unittest.main()
