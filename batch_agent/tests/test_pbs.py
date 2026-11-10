"""
Unit tests for PBS / OpenPBS / PBS Pro environment variable detection and job handling.
"""

import os
import tempfile
import unittest

from batch_agent.core.job_spec import BatchJobSpec


class TestPBSIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)
        self.original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)
        self.tmp_dir.cleanup()

    def test_pbs_environment_detection(self):
        # Set PBS environment variables
        os.environ["PBS_JOBID"] = "987654.pbs-headnode.cluster.local"
        os.environ["PBS_O_WORKDIR"] = self.ws
        os.environ["PBS_QUEUE"] = "workq"

        config = BatchJobSpec.from_dict({
            "goal": "Execute PBS batch job",
        })

        # Check job_id stripped to 987654
        self.assertEqual(config.job_id, "987654")
        self.assertEqual(config.workspace_root, self.ws)

    def test_pbs_template_file_exists_and_valid(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "batch_agent", "templates", "pbs_job.pbs"
        )
        self.assertTrue(os.path.exists(template_path), "pbs_job.pbs template must exist")

        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("#PBS -N", content)
        self.assertIn("#PBS -l", content)
        self.assertIn("PBS_O_WORKDIR", content)
        self.assertIn("batch_agent.cli", content)


if __name__ == "__main__":
    unittest.main()

