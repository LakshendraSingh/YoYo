"""
Direct Model Weights Inference Provider for offline batch HPC nodes and GPU clusters.
Supports direct loading of:
1. GGUF / GGML quantized weights via llama-cpp-python or standalone llama-cli
2. HuggingFace / Safetensors / PyTorch weights via transformers & torch
3. Fallback to resilient heuristic engine with audit logging if drivers/backends unavailable.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from batch_agent.config import BatchAgentConfig
from batch_agent.intelligence.cot_engine import ChainOfThoughtEngine, ChainOfThoughtRecord
from batch_agent.intelligence.planner import ExecutionPlan
from batch_agent.intelligence.provider import AgentDecision, BaseProvider, HeuristicAutonomousProvider

logger = logging.getLogger(__name__)


class LocalWeightsProvider(BaseProvider):
    """
    Connects directly to raw model weights stored on local disk or cluster shared storage
    (e.g. /models/llama-3-8b.gguf, /shared/weights/deepseek-coder-safetensors).
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        n_gpu_layers: int = -1,
        context_window: int = 4096,
        temperature: float = 0.2,
    ) -> None:
        self.weights_path = os.path.realpath(os.path.abspath(weights_path))
        self.device = device
        self.n_gpu_layers = n_gpu_layers
        self.context_window = context_window
        self.temperature = temperature
        self.fallback = HeuristicAutonomousProvider()

        # Cached engine instances
        self._backend_type: Optional[str] = None
        self._llama_model = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._llama_cli_path: Optional[str] = None

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Detect weights format and initialize the best available inference backend."""
        if not os.path.exists(self.weights_path):
            logger.warning("Specified model weights path does not exist: %s", self.weights_path)
            self._backend_type = "unavailable"
            return

        is_gguf = (
            os.path.isfile(self.weights_path)
            and self.weights_path.lower().endswith((".gguf", ".bin", ".q4_k_m", ".q8_0"))
        )
        is_hf_dir = os.path.isdir(self.weights_path) and any(
            os.path.exists(os.path.join(self.weights_path, f))
            for f in ("config.json", "model.safetensors", "pytorch_model.bin")
        )
        is_safetensors_file = os.path.isfile(self.weights_path) and self.weights_path.lower().endswith(".safetensors")

        # 1. Try llama-cpp-python for GGUF
        if is_gguf:
            try:
                import llama_cpp
                logger.info("Initializing llama-cpp-python with weights: %s", self.weights_path)
                self._llama_model = llama_cpp.Llama(
                    model_path=self.weights_path,
                    n_gpu_layers=self.n_gpu_layers,
                    n_ctx=self.context_window,
                    verbose=False,
                )
                self._backend_type = "llama_cpp"
                return
            except ImportError:
                pass
            except Exception as e:
                logger.warning("llama-cpp-python failed to load weights: %s", e)

            # 2. Check for local llama-cli binary
            cli_bin = shutil.which("llama-cli") or shutil.which("llama.cpp") or shutil.which("main")
            for search_path in ("/usr/local/bin/llama-cli", "/opt/llama.cpp/llama-cli", os.path.expanduser("~/llama.cpp/llama-cli")):
                if not cli_bin and os.path.isfile(search_path) and os.access(search_path, os.X_OK):
                    cli_bin = search_path

            if cli_bin:
                self._llama_cli_path = cli_bin
                self._backend_type = "llama_cli"
                logger.info("Using standalone llama-cli binary: %s", cli_bin)
                return

        # 3. Try HuggingFace Transformers for directory or safetensors
        if is_hf_dir or is_safetensors_file:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Initializing Transformers with weights: %s", self.weights_path)
                model_dir = os.path.dirname(self.weights_path) if is_safetensors_file else self.weights_path
                self._hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)

                self._hf_model_dir = model_dir
                self._load_transformers_model(torch, AutoModelForCausalLM, model_dir, headroom_ratio=0.80)
                self._backend_type = "transformers"
                return
            except ImportError:
                pass
            except Exception as e:
                logger.warning("Transformers failed to load model weights: %s", e)

        # If no native runtime backend is installed on this node
        logger.info(
            "Model weights detected at '%s', but no native weight inference runtime (llama-cpp-python, transformers, or llama-cli) is installed. Heuristic autonomous planner will be used.",
            self.weights_path,
        )
        self._backend_type = "heuristic_fallback"

    @property
    def backend_type(self) -> str:
        return self._backend_type or "uninitialized"

    @staticmethod
    def _build_cuda_max_memory(torch_mod: Any, headroom_ratio: float = 0.80) -> Optional[Dict[Any, str]]:
        """
        Build per-GPU max_memory limits leaving VRAM headroom for KV cache & activations.
        Prevents accelerate/device_map='auto' from packing 100% of GPU 0 on multi-GPU nodes (e.g. 4x H100).
        """
        if not torch_mod.cuda.is_available():
            return None
        num_gpus = torch_mod.cuda.device_count()
        if num_gpus <= 0:
            return None
        max_memory: Dict[Any, str] = {}
        for idx in range(num_gpus):
            try:
                free_bytes, _ = torch_mod.cuda.mem_get_info(idx)
            except Exception:
                free_bytes = torch_mod.cuda.get_device_properties(idx).total_memory
            # Reserve extra headroom on GPU 0 where input tensors & LM head logits allocate
            gpu_ratio = headroom_ratio if idx == 0 else min(0.90, headroom_ratio + 0.08)
            usable_gib = max(1, int((free_bytes * gpu_ratio) / (1024 ** 3)))
            max_memory[idx] = f"{usable_gib}GiB"
        max_memory["cpu"] = "128GiB"
        return max_memory

    def _load_transformers_model(self, torch_mod: Any, auto_model_cls: Any, model_dir: str, headroom_ratio: float = 0.80) -> None:
        """Load HuggingFace model sharded across all available GPUs with KV-cache headroom."""
        use_cuda = self.device in ("auto", "cuda") and torch_mod.cuda.is_available()
        device_map = "auto" if use_cuda else None
        max_memory = self._build_cuda_max_memory(torch_mod, headroom_ratio=headroom_ratio) if use_cuda else None

        kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if max_memory is not None:
            kwargs["max_memory"] = max_memory

        self._hf_model = auto_model_cls.from_pretrained(model_dir, **kwargs)

    def _run_inference(self, prompt: str) -> str:
        """Execute a single inference pass on the active local weights backend."""
        if self._backend_type == "llama_cpp" and self._llama_model is not None:
            response = self._llama_model(
                prompt=prompt,
                max_tokens=1024,
                temperature=self.temperature,
                stop=["</s>", "<|im_end|>", "\n\n\n"],
            )
            return response["choices"][0]["text"]

        if self._backend_type == "llama_cli" and self._llama_cli_path is not None:
            cmd = [
                self._llama_cli_path,
                "--model", self.weights_path,
                "--prompt", prompt,
                "--ctx-size", str(self.context_window),
                "--n-gpu-layers", str(self.n_gpu_layers),
                "--n-predict", "1024",
                "--temp", str(self.temperature),
                "--threads", str(min(8, os.cpu_count() or 4)),
            ]
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0 and "out of memory" in (proc.stderr or "").lower():
                raise RuntimeError(f"llama-cli out of memory: {proc.stderr.strip()}")
            return proc.stdout

        if self._backend_type == "transformers" and self._hf_model is not None:
            import torch
            inputs = self._hf_tokenizer(prompt, return_tensors="pt")
            if hasattr(self._hf_model, "device"):
                inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                )
            return self._hf_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return ""

    def _recover_from_oom(self) -> bool:
        """Free GPU cache and reload model with larger KV-cache VRAM headroom."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if self._backend_type == "transformers" and getattr(self, "_hf_model_dir", None):
            try:
                import torch
                from transformers import AutoModelForCausalLM
                self._hf_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("VRAM OOM during inference: reloading Transformers model with 35%% KV-cache headroom...")
                self._load_transformers_model(torch, AutoModelForCausalLM, self._hf_model_dir, headroom_ratio=0.65)
                return True
            except Exception as reload_err:
                logger.warning("Transformers OOM reload failed: %s", reload_err)
                return False

        if self._backend_type == "llama_cpp" and self._llama_model is not None:
            try:
                import llama_cpp
                self._llama_model = None
                gc.collect()
                next_layers = 40 if self.n_gpu_layers == -1 else max(0, self.n_gpu_layers // 2)
                if next_layers == self.n_gpu_layers:
                    return False
                self.n_gpu_layers = next_layers
                logger.warning("VRAM OOM during llama.cpp inference: reloading with n_gpu_layers=%d...", next_layers)
                self._llama_model = llama_cpp.Llama(
                    model_path=self.weights_path,
                    n_gpu_layers=next_layers,
                    n_ctx=self.context_window,
                    verbose=False,
                )
                return True
            except Exception as reload_err:
                logger.warning("llama.cpp OOM reload failed: %s", reload_err)
                return False

        if self._backend_type == "llama_cli":
            next_layers = 40 if self.n_gpu_layers == -1 else max(0, self.n_gpu_layers // 2)
            if next_layers != self.n_gpu_layers:
                self.n_gpu_layers = next_layers
                return True

        return False

    def decide_next_step(
        self,
        config: BatchAgentConfig,
        plan: ExecutionPlan,
        step_number: int,
        observation: str,
        last_error: Optional[str] = None,
        workspace_files: Optional[List[str]] = None,
    ) -> AgentDecision:
        # If weights failed to load or runtime missing, use resilient fallback
        if self._backend_type in ("unavailable", "heuristic_fallback", None):
            decision = self.fallback.decide_next_step(config, plan, step_number, observation, last_error, workspace_files)
            # Annotate reasoning with weights note
            decision.thought.hypotheses.append(f"Model weights path configured: {self.weights_path} (Backend: {self.backend_type})")
            return decision

        prompt = self._format_prompt(config, plan, step_number, observation, last_error)

        raw_output = ""
        try:
            try:
                raw_output = self._run_inference(prompt)
            except Exception as infer_err:
                if "out of memory" in str(infer_err).lower() and self._recover_from_oom():
                    raw_output = self._run_inference(prompt)
                else:
                    raise

            # Parse JSON output from model
            parsed = self._extract_json(raw_output)
            if parsed:
                thought = ChainOfThoughtEngine.create_thought(
                    phase=parsed.get("phase", "ACTING"),
                    goal_analysis=config.goal,
                    observation=observation[:200],
                    decision=parsed.get("decision", ""),
                    tradeoffs=[parsed.get("tradeoffs", "")] if parsed.get("tradeoffs") else [],
                )
                return AgentDecision(
                    thought=thought,
                    tool_name=parsed.get("tool_name"),
                    tool_params=parsed.get("tool_params", {}),
                    is_finish=parsed.get("is_finish", False),
                    finish_summary=parsed.get("finish_summary", ""),
                )
        except Exception as e:
            logger.warning("Error during direct model weights inference: %s", e)

        # Resilient fallback on inference parse error
        return self.fallback.decide_next_step(config, plan, step_number, observation, last_error, workspace_files)

    def _format_prompt(
        self,
        config: BatchAgentConfig,
        plan: ExecutionPlan,
        step_number: int,
        observation: str,
        last_error: Optional[str],
    ) -> str:
        return (
            "<|im_start|>system\n"
            "You are an autonomous AI operating in a NON-INTERACTIVE BATCH OS.\n"
            "You have direct access to system tools. Make optimal decisions end-to-end.\n"
            "Respond ONLY with a valid JSON object matching:\n"
            '{"phase": "PLANNING|ACTING|REFLECTING|VERIFYING", "reasoning": "...", "tradeoffs": "...", "decision": "...", "tool_name": "...", "tool_params": {}, "is_finish": false, "finish_summary": ""}\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Goal: {config.goal}\n"
            f"Associated Files: {config.associated_files}\n"
            f"Step: {step_number}\n"
            f"Observation: {observation}\n"
            f"Last Error: {last_error or 'None'}\n"
            f"Current Plan:\n{plan.format_summary()}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse first valid JSON block from raw model output."""
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

