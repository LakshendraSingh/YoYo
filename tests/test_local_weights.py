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

    def test_multi_gpu_h100_max_memory_reserves_kv_headroom(self):
        class MockCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 4

            @staticmethod
            def mem_get_info(idx):
                h100_80g = 80 * (1024 ** 3)
                return (h100_80g, h100_80g)

        class MockTorch:
            cuda = MockCuda()

        mem_map = LocalWeightsProvider._build_cuda_max_memory(MockTorch(), headroom_ratio=0.80)
        self.assertEqual(mem_map[0], "64GiB")
        self.assertEqual(mem_map[1], "70GiB")
        self.assertEqual(mem_map[2], "70GiB")
        self.assertEqual(mem_map[3], "70GiB")
        self.assertEqual(mem_map["cpu"], "128GiB")

    def test_oom_during_inference_triggers_recovery_and_retry(self):
        non_existent = os.path.join(self.ws, "missing.gguf")
        provider = LocalWeightsProvider(weights_path=non_existent)
        provider._backend_type = "transformers"

        calls = {"count": 0, "recovered": False}

        def fake_infer(prompt):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return '{"phase": "ACTING", "decision": "read file", "tool_name": "file_read", "tool_params": {"path": "a.py"}, "is_finish": false}'

        def fake_recover():
            calls["recovered"] = True
            return True

        provider._run_inference = fake_infer  # type: ignore
        provider._recover_from_oom = fake_recover  # type: ignore

        plan = ExecutionPlan(goal="Test OOM recovery")
        config = BatchAgentConfig(goal="Test OOM recovery", workspace_root=self.ws)
        decision = provider.decide_next_step(config, plan, step_number=1, observation="Start")

        self.assertTrue(calls["recovered"])
        self.assertEqual(calls["count"], 2)
        self.assertEqual(decision.tool_name, "file_read")
        self.assertEqual(decision.tool_params, {"path": "a.py"})


if __name__ == "__main__":
    unittest.main()

