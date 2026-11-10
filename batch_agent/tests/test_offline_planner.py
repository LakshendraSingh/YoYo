"""
Unit and integration tests for offline planning, dependency exploration,
and plan.txt manifest generation and resumption across batch cycles.
"""

import os
import tempfile
import unittest

from batch_agent.cli import main
from batch_agent.intelligence.dependency_planner import DependencyPlanner
from batch_agent.tools.env_explorer import EnvironmentProbe


class TestOfflinePlanner(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_environment_probe_installed_and_binaries(self):
        inv = EnvironmentProbe.probe(self.ws)
        self.assertTrue(len(inv.python_version) > 0)
        self.assertTrue(os.path.exists(inv.python_path))
        self.assertIn("python3", inv.system_binaries)

    def test_workspace_import_detection_identifies_missing(self):
        # Create a Python file importing an uninstalled package
        sample_code = (
            "import os\n"
            "import sys\n"
            "import nonexistent_custom_library_xyz\n"
            "from imaginary_data_pkg import compute\n"
        )
        src_dir = os.path.join(self.ws, "src")
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, "analysis.py"), "w") as f:
            f.write(sample_code)

        inv = EnvironmentProbe.probe(self.ws)
        self.assertIn("nonexistent_custom_library_xyz", inv.missing_packages)
        self.assertIn("imaginary_data_pkg", inv.missing_packages)

    def test_generate_plan_file_with_missing_dependencies(self):
        # Create script with missing dependency
        with open(os.path.join(self.ws, "pipeline.py"), "w") as f:
            f.write("import torch_custom_vision_nonexistent\n")

        plan_path, inv, has_missing = DependencyPlanner.generate_plan_file(
            goal="Train custom vision model",
            workspace_root=self.ws,
            associated_files=["pipeline.py"],
            output_filename="plan.txt",
        )

        self.assertTrue(has_missing)
        self.assertTrue(os.path.exists(plan_path))

        with open(plan_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("YOYO - OFFLINE EXECUTION PLAN & DEPENDENCY MANIFEST", content)
        self.assertIn("[SECTION 1: ENVIRONMENT INVENTORY]", content)
        self.assertIn("[MISSING] torch_custom_vision_nonexistent", content)
        self.assertIn("[SECTION 3: DETAILED DEPENDENCY RATIONALE & NEXT ACTIONS]", content)
        self.assertIn("Why Needed", content)
        self.assertIn("Next Action", content)
        self.assertIn("[SECTION 4: PROPOSED ARCHITECTURE & CODE IN proposal/]", content)
        self.assertIn("pip download", content)
        self.assertIn("yoyo --plan plan.txt", content)

        # Verify proposal/ directory and drafted programs exist
        proposal_solution = os.path.join(self.ws, "proposal", "solution.py")
        proposal_test = os.path.join(self.ws, "proposal", "test_solution.py")
        self.assertTrue(os.path.exists(proposal_solution), "proposal/solution.py must be drafted")
        self.assertTrue(os.path.exists(proposal_test), "proposal/test_solution.py must be drafted")

    def test_load_plan_file(self):
        sample_plan = (
            "================================================================================\n"
            "YOYO - OFFLINE EXECUTION PLAN & DEPENDENCY MANIFEST\n"
            "================================================================================\n"
            "Primary Goal        : Execute sample telemetry pipeline\n"
            "Network Mode        : OFFLINE / AIR-GAPPED (NO INTERNET)\n"
            "\n"
            "Missing Required Dependencies (Action Needed by User/Admin):\n"
            "  [MISSING] pandas               -> Required by codebase\n"
            "  [MISSING] scipy                -> Required by codebase\n"
            "\n"
            "[SECTION 4: EXECUTION SUBGOALS]\n"
            "  1. [PENDING] Inspect input file data/input.csv\n"
            "  2. [PENDING] Execute data processing script\n"
            "  3. [PENDING] Run verification unit test suite\n"
            "================================================================================\n"
        )
        plan_file = os.path.join(self.ws, "plan.txt")
        with open(plan_file, "w") as f:
            f.write(sample_plan)

        loaded = DependencyPlanner.load_plan_file(plan_file)
        self.assertEqual(loaded.goal, "Execute sample telemetry pipeline")
        self.assertIn("pandas", loaded.missing_dependencies)
        self.assertIn("scipy", loaded.missing_dependencies)
        self.assertEqual(len(loaded.steps), 3)

        # Convert to execution plan
        exec_plan = loaded.to_execution_plan()
        self.assertEqual(len(exec_plan.steps), 3)
        self.assertEqual(exec_plan.steps[0].action_type, "inspect")
        self.assertEqual(exec_plan.steps[1].action_type, "execute")
        self.assertEqual(exec_plan.steps[2].action_type, "verify")

    def test_two_cycle_batch_workflow_with_plan_txt(self):
        # Cycle 1: Run with --no-internet on a workspace needing external package
        script_file = os.path.join(self.ws, "job.py")
        with open(script_file, "w") as f:
            f.write("import missing_offline_lib\n")

        audit_1 = os.path.join(self.ws, "audit_1.txt")
        exit_code_1 = main([
            "--goal", "Run job with missing offline library",
            "--workspace", self.ws,
            "--no-internet",
            "--log-file", audit_1,
        ])

        # Cycle 1 generates plan.txt and exits 0 cleanly informing user
        self.assertEqual(exit_code_1, 0)
        plan_file = os.path.join(self.ws, "plan.txt")
        self.assertTrue(os.path.exists(plan_file))
        with open(plan_file, "r") as f:
            plan_text = f.read()
        self.assertIn("missing_offline_lib", plan_text)

        # Cycle 2: User provides plan.txt for the next batch run
        audit_2 = os.path.join(self.ws, "audit_2.txt")
        exit_code_2 = main([
            "--plan", plan_file,
            "--workspace", self.ws,
            "--no-internet",
            "--log-file", audit_2,
        ])

        self.assertEqual(exit_code_2, 0)
        self.assertTrue(os.path.exists(audit_2))
        with open(audit_2, "r") as f:
            audit_2_text = f.read()
        self.assertIn("Offline Plan Ingested from plan.txt", audit_2_text)


if __name__ == "__main__":
    unittest.main()

