"""
Unit tests for CLI parsing and manifest ingestion.
"""

import json
import os
import tempfile
import unittest

from batch_agent.cli import create_parser, main


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_parser_with_goal(self):
        parser = create_parser()
        args = parser.parse_args(["--goal", "Run regression suite", "--workspace", self.ws])
        self.assertEqual(args.goal, "Run regression suite")
        self.assertEqual(args.workspace, self.ws)

    def test_cli_execution_via_goal(self):
        log_file = os.path.join(self.ws, "cli_audit.txt")
        exit_code = main([
            "--goal", "Initialize workspace and report status",
            "--workspace", self.ws,
            "--log-file", log_file,
            "--max-steps", "5",
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(log_file))

    def test_cli_execution_via_manifest(self):
        manifest_data = {
            "job_id": "cli-manifest-test",
            "goal": "Execute manifest defined goal",
            "workspace_root": self.ws,
            "max_steps": 5,
            "audit": {
                "log_file_path": os.path.join(self.ws, "manifest_audit.txt"),
            },
        }
        manifest_file = os.path.join(self.ws, "job.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f)

        exit_code = main(["--manifest", manifest_file])
        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(os.path.join(self.ws, "manifest_audit.txt")))


if __name__ == "__main__":
    unittest.main()

