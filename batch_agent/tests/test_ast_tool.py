"""
Unit tests for the CodeASTInspectTool.
"""

import os
import tempfile
import unittest

from batch_agent.config import ProcessConfig, SandboxConfig
from batch_agent.process.runner import ProcessRunner
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager
from batch_agent.tools.ast_tool import CodeASTInspectTool
from batch_agent.tools.base import ToolContext


class TestASTTool(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.sandbox = SandboxManager(SandboxConfig(workspace_root=self.ws), ProcessConfig())
        self.tracker = ProcessTracker()
        self.runner = ProcessRunner(self.sandbox, self.tracker)
        self.context = ToolContext(
            workspace_root=self.ws,
            cwd=self.ws,
            sandbox=self.sandbox,
            tracker=self.tracker,
            runner=self.runner,
        )
        self.tool = CodeASTInspectTool()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_ast_inspect_symbols(self):
        code = (
            "import os\n"
            "from math import sqrt, pi\n\n"
            "class Circle:\n"
            "    '''Geometric circle.'''\n"
            "    def __init__(self, radius: float):\n"
            "        self.radius = radius\n"
            "    def area(self) -> float:\n"
            "        return pi * (self.radius ** 2)\n\n"
            "def calculate_hypotenuse(a: float, b: float) -> float:\n"
            "    '''Compute triangle hypotenuse.'''\n"
            "    return sqrt(a**2 + b**2)\n"
        )
        filepath = os.path.join(self.ws, "math_utils.py")
        with open(filepath, "w") as f:
            f.write(code)

        result = self.tool.run({"path": "math_utils.py"}, self.context)
        self.assertTrue(result.success)
        self.assertIn("class Circle:", result.output)
        self.assertIn("def __init__(self, radius):", result.output)
        self.assertIn("def calculate_hypotenuse(a, b):", result.output)
        self.assertIn("from math import sqrt, pi", result.output)

    def test_ast_inspect_syntax_error(self):
        broken_code = "def broken(:\n    pass\n"
        filepath = os.path.join(self.ws, "broken.py")
        with open(filepath, "w") as f:
            f.write(broken_code)

        result = self.tool.run({"path": "broken.py"}, self.context)
        self.assertFalse(result.success)
        self.assertIn("Syntax error at line 1", result.error)


if __name__ == "__main__":
    unittest.main()

