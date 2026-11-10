"""
Unit tests for direct local model weights loading and inference provider.
"""

import os
import tempfile
import unittest

from batch_agent.cli import create_parser
from batch_agent.config import BatchAgentConfig, LLMProviderType
from batch_agent.intelligence.local_weights import LocalWeightsProvider
from batch_agent.intelligence.planner import ExecutionPlan


class TestLocalWeightsProvider(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws = os.path.realpath(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_cli_parser_model_weights_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            "--goal", "Analyze data with local weights",
            "--model-weights", "/models/llama-3-8b.gguf",
            "--device", "cuda",
            "--gpu-layers", "33",
            "--context-window", "8192",
        ])
        self.assertEqual(args.model_weights, "/models/llama-3-8b.gguf")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.gpu_layers, 33)
        self.assertEqual(args.context_window, 8192)

    def test_local_weights_provider_nonexistent_path(self):
        non_existent = os.path.join(self.ws, "missing_weights.gguf")
        provider = LocalWeightsProvider(weights_path=non_existent)
        self.assertEqual(provider.backend_type, "unavailable")

        # Must fall back gracefully to heuristic planner rather than crashing
        plan = ExecutionPlan(goal="Test goal")
        config = BatchAgentConfig(goal="Test goal", workspace_root=self.ws)
        decision = provider.decide_next_step(config, plan, step_number=1, observation="Ready")
        self.assertIsNotNone(decision.thought)
        self.assertIsNotNone(decision.tool_name)

    def test_local_weights_provider_gguf_detection(self):
        # Create dummy GGUF weights file
        dummy_gguf = os.path.join(self.ws, "model.gguf")
        with open(dummy_gguf, "wb") as f:
            f.write(b"GGUF" + b"\x00" * 64)

        provider = LocalWeightsProvider(weights_path=dummy_gguf)
        # In a test environment without llama-cpp installed, it should recognize the file and set fallback
        self.assertIn(provider.backend_type, ["llama_cpp", "llama_cli", "heuristic_fallback"])

        plan = ExecutionPlan(goal="Run local batch inference")
        config = BatchAgentConfig(goal="Run local batch inference", workspace_root=self.ws)
        decision = provider.decide_next_step(config, plan, step_number=1, observation="Start")
        self.assertIsNotNone(decision.thought)

    def test_local_weights_provider_hf_dir_detection(self):
        # Create dummy HuggingFace model directory
        hf_dir = os.path.join(self.ws, "hf_model")
        os.makedirs(hf_dir, exist_ok=True)
        with open(os.path.join(hf_dir, "config.json"), "w") as f:
            f.write('{"model_type": "llama", "hidden_size": 4096}')

        provider = LocalWeightsProvider(weights_path=hf_dir)
        self.assertIn(provider.backend_type, ["transformers", "heuristic_fallback"])


if __name__ == "__main__":
    unittest.main()

