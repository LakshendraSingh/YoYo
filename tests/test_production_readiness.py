"""Unit tests for real-world production readiness: .gitignore/venv filtering, stub safety, and working memory."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from batch_agent.config import BatchAgentConfig, ProcessConfig, SandboxConfig
from batch_agent.intelligence.cot_engine import ChainOfThoughtEngine
from batch_agent.intelligence.provider import AgentDecision, GeminiProvider
from batch_agent.process.recovery import CommandRecoveryEngine, FailureCategory
from batch_agent.process.runner import ProcessResult, ProcessRunner
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager
from batch_agent.tools.ast_tool import CodeASTInspectTool
from batch_agent.tools.base import ToolContext
from batch_agent.tools.filesystem import FileEditTool, FileListTool, FileSearchTool


class TestProductionReadiness(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.sandbox = SandboxManager(SandboxConfig(workspace_root=self.ws), ProcessConfig())
        self.tracker = ProcessTracker()
        self.runner = ProcessRunner(sandbox=self.sandbox, tracker=self.tracker, config=ProcessConfig())

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_context(self, engine=None):
        return ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=self.sandbox,
            tracker=self.tracker,
            runner=self.runner,
            engine=engine,
        )

    def test_gitignore_and_venv_exclusion_in_filesystem_tools(self):
        # Create noisy real-world directories that should be pruned
        for rel_dir in (".venv/lib/python3.11", "node_modules/lodash", "dist", "build", "generated"):
            os.makedirs(os.path.join(self.ws, rel_dir), exist_ok=True)

        with open(os.path.join(self.ws, ".gitignore"), "w", encoding="utf-8") as f:
            f.write("*.log\ngenerated/\n")

        with open(os.path.join(self.ws, ".venv/lib/python3.11/app.py"), "w", encoding="utf-8") as f:
            f.write("def target_symbol():\n    return 'venv_noise'\n")
        with open(os.path.join(self.ws, "node_modules/lodash/index.js"), "w", encoding="utf-8") as f:
            f.write("function target_symbol() {}\n")
        with open(os.path.join(self.ws, "generated/auto.py"), "w", encoding="utf-8") as f:
            f.write("def target_symbol():\n    return 'ignored'\n")
        with open(os.path.join(self.ws, "debug.log"), "w", encoding="utf-8") as f:
            f.write("target_symbol error\n")

        src_dir = os.path.join(self.ws, "src")
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write("def target_symbol():\n    return 'real_source'\n")

        ctx = self._make_context()

        # 1. FileListTool (recursive) should only show src/app.py
        list_res = FileListTool().run({"recursive": True}, ctx)
        self.assertTrue(list_res.success)
        self.assertIn("src/app.py", list_res.output)
        self.assertNotIn(".venv", list_res.output)
        self.assertNotIn("node_modules", list_res.output)
        self.assertNotIn("generated", list_res.output)
        self.assertNotIn("debug.log", list_res.output)

        # 2. FileSearchTool should only match src/app.py
        search_res = FileSearchTool().run({"pattern": "target_symbol"}, ctx)
        self.assertTrue(search_res.success)
        self.assertIn("src/app.py:1:", search_res.output)
        self.assertEqual(search_res.metadata["match_count"], 1)

        # 3. FileEditTool shorthand path ('app.py') should not collide with .venv/.../app.py
        edit_res = FileEditTool().run(
            {"path": "app.py", "target": "return 'real_source'", "replacement": "return 'patched'"},
            ctx,
        )
        self.assertTrue(edit_res.success, edit_res.error)
        with open(os.path.join(src_dir, "app.py"), "r", encoding="utf-8") as f:
            self.assertIn("return 'patched'", f.read())

    def test_dummy_stubs_disabled_by_default(self):
        class MockEngine:
            config = BatchAgentConfig(workspace_root=self.ws, allow_dummy_stubs=False)

        ctx = self._make_context(engine=MockEngine())
        failed_res = ProcessResult(
            command=f"{sys.executable} -c 'import missing_prod_pkg'",
            exit_code=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'missing_prod_pkg'",
            duration_seconds=0.05,
        )
        attempt = CommandRecoveryEngine.attempt_recovery(
            command=failed_res.command,
            result=failed_res,
            context=ctx,
        )
        self.assertFalse(attempt.success)
        self.assertFalse(os.path.exists(os.path.join(self.ws, "missing_prod_pkg.py")))

    def test_working_memory_tracks_cross_step_history(self):
        provider = GeminiProvider(api_key="dummy-key")
        thought = ChainOfThoughtEngine.create_thought("ACTING", "fix bug", "obs", "read file")

        # Step 1 decision
        h1 = provider._update_and_format_memory(1, "Initial state", None)
        self.assertEqual(h1, "None (First Step)")
        provider._remember_decision(
            1,
            AgentDecision(thought=thought, tool_name="file_read", tool_params={"path": "src/auth.py"}),
        )

        # Step 2 sees Step 1 outcome in working memory
        h2 = provider._update_and_format_memory(2, "def verify_token(): return False", None)
        self.assertIn("Step 1: file_read", h2)
        self.assertIn("def verify_token()", h2)
        provider._remember_decision(
            2,
            AgentDecision(thought=thought, tool_name="file_read", tool_params={"path": "src/routes.py"}),
        )

        # Step 3 retains both Step 1 and Step 2
        h3 = provider._update_and_format_memory(3, "app.route('/login')", None)
        self.assertIn("Step 1: file_read", h3)
        self.assertIn("Step 2: file_read", h3)
        self.assertIn("app.route('/login')", h3)

    def test_cpp_and_polyglot_test_failures_not_intercepted(self):
        samples = [
            ("ctest --test-dir build", "1/2 Test #1: MathTest .........................***Failed    0.01 sec"),
            ("./build/unit_tests", "[  FAILED  ] MathSuite.MultiplyTest (0 ms)"),
            ("./test_bin", "Assertion failed: (result == 42), function main, file main.c, line 12."),
            ("go test ./...", "--- FAIL: TestMultiply (0.00s)"),
        ]
        ctx = self._make_context()
        for cmd, out in samples:
            res = ProcessResult(command=cmd, exit_code=1, stdout=out, stderr="", duration_seconds=0.05)
            cat, _ = CommandRecoveryEngine.diagnose_failure(cmd, res)
            self.assertEqual(cat, FailureCategory.ASSERTION_OR_TEST_FAILURE, f"Failed for: {cmd}")
            attempt = CommandRecoveryEngine.attempt_recovery(cmd, res, ctx)
            self.assertFalse(attempt.success)
            self.assertIsNone(attempt.repaired_command)

    def test_ast_inspect_supports_c_and_cpp(self):
        cpp_path = os.path.join(self.ws, "engine.cpp")
        cpp_code = """#include <vector>
#include "analytics.hpp"

// void fake_commented_func() {}
/*
class FakeCommentedClass {
    int ghost() { return 0; }
};
*/

struct MetricPoint {
    double value;
};

class AnalyticsEngine {
public:
    AnalyticsEngine(int scale) {}
    int multiply(int a, int b) const {
        if (a == 0) return 0;
        return a * b;
    }
    ~AnalyticsEngine() {}
};

int compute_sum(int x, int y) {
    return x + y;
}
"""
        with open(cpp_path, "w", encoding="utf-8") as f:
            f.write(cpp_code)

        res = CodeASTInspectTool().run({"path": "engine.cpp"}, self._make_context())
        self.assertTrue(res.success, res.error)
        self.assertIn("#include <vector>", res.output)
        self.assertIn('#include "analytics.hpp"', res.output)
        self.assertIn("struct MetricPoint:", res.output)
        self.assertIn("class AnalyticsEngine:", res.output)
        self.assertIn("int multiply(int a, int b)", res.output)
        self.assertIn("int compute_sum(int x, int y)", res.output)
        self.assertNotIn("fake_commented_func", res.output)
        self.assertNotIn("FakeCommentedClass", res.output)
        self.assertNotIn("if (", res.output)


if __name__ == "__main__":
    unittest.main()
