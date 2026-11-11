"""
Unit tests for filesystem tools, resilient multi-tier fuzzy patching, and verification tools.
"""

import os
import tempfile
import unittest

from batch_agent.config import ProcessConfig, SandboxConfig
from batch_agent.process.runner import ProcessRunner
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager
from batch_agent.tools.base import ToolContext
from batch_agent.tools.filesystem import (
    FileEditTool,
    FileListTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
)
from batch_agent.tools.registry import create_default_registry
from batch_agent.tools.verification import VerifyCodeTool


class TestTools(unittest.TestCase):
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
            step_number=1,
        )
        self.registry = create_default_registry()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_file_write_and_read(self):
        write_res = self.registry.execute(
            "file_write",
            {"path": "hello.txt", "content": "Hello Batch World!\nLine 2"},
            self.context,
        )
        self.assertTrue(write_res.success)

        read_res = self.registry.execute(
            "file_read",
            {"path": "hello.txt"},
            self.context,
        )
        self.assertTrue(read_res.success)
        self.assertIn("Hello Batch World!", read_res.output)
        self.assertIn("Line 2", read_res.output)

    def test_file_edit_resilient_patching(self):
        initial_code = (
            "def calculate(a, b):\n"
            "    # Old logic\n"
            "    result = a + b\n"
            "    return result\n"
        )
        filepath = os.path.join(self.ws, "calc.py")
        with open(filepath, "w") as f:
            f.write(initial_code)

        # 1. Exact match patch
        patch_res = self.registry.execute(
            "file_edit",
            {
                "path": "calc.py",
                "target": "    # Old logic\n    result = a + b",
                "replacement": "    # Enhanced logic\n    result = a * b",
            },
            self.context,
        )
        self.assertTrue(patch_res.success)
        with open(filepath, "r") as f:
            updated = f.read()
        self.assertIn("# Enhanced logic", updated)
        self.assertIn("result = a * b", updated)

        # 2. Fuzzy / whitespace-tolerant patch
        fuzzy_patch_res = self.registry.execute(
            "file_edit",
            {
                "path": "calc.py",
                "target": "# Enhanced logic\nresult = a * b",  # Missing indentation
                "replacement": "    # Resilient patch succeeded\n    result = a ** b",
            },
            self.context,
        )
        self.assertTrue(fuzzy_patch_res.success)
        with open(filepath, "r") as f:
            final_code = f.read()
        self.assertIn("Resilient patch succeeded", final_code)

    def test_file_list_and_search(self):
        os.makedirs(os.path.join(self.ws, "module"), exist_ok=True)
        with open(os.path.join(self.ws, "module", "worker.py"), "w") as f:
            f.write("def do_work():\n    return 'YOYO_TOKEN'\n")

        list_res = self.registry.execute("file_list", {"recursive": True}, self.context)
        self.assertTrue(list_res.success)
        self.assertIn("module/worker.py", list_res.output)

        search_res = self.registry.execute(
            "file_search",
            {"pattern": "YOYO_TOKEN"},
            self.context,
        )
        self.assertTrue(search_res.success)
        self.assertIn("worker.py", search_res.output)

    def test_verify_code_syntax(self):
        good_file = os.path.join(self.ws, "good.py")
        with open(good_file, "w") as f:
            f.write("print('valid syntax')\n")

        bad_file = os.path.join(self.ws, "bad.py")
        with open(bad_file, "w") as f:
            f.write("def broken(:\n    pass\n")

        verify_good = self.registry.execute(
            "verify_code",
            {"mode": "syntax", "target": "good.py"},
            self.context,
        )
        self.assertTrue(verify_good.success)

        verify_bad = self.registry.execute(
            "verify_code",
            {"mode": "syntax", "target": "bad.py"},
            self.context,
        )
        self.assertFalse(verify_bad.success)


if __name__ == "__main__":
    unittest.main()

