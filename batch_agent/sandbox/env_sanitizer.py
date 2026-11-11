"""
Subprocess environment sanitizer for scrubbed, non-interactive execution.
Strips credentials and secrets to prevent token exfiltration and accidental leaks.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from batch_agent.config import ProcessConfig, SandboxConfig


class EnvironmentSanitizer:
    """Sanitizes environment dictionaries for spawned subprocesses in batch jobs."""

    def __init__(self, sandbox_config: SandboxConfig, process_config: ProcessConfig) -> None:
        self.sandbox_config = sandbox_config
        self.process_config = process_config

    def sanitize(self, base_env: Dict[str, str] | None = None) -> Tuple[Dict[str, str], List[str]]:
        """
        Produce a scrubbed environment dictionary for child subprocesses.
        Returns (sanitized_env, list_of_scrubbed_variable_names).
        """
        raw_env = os.environ.copy() if base_env is None else base_env.copy()
        scrubbed_keys: List[str] = []

        if self.sandbox_config.scrub_environment:
            # 1. Exact matches
            for key in self.sandbox_config.scrubbed_env_vars:
                if key in raw_env:
                    del raw_env[key]
                    scrubbed_keys.append(key)

            # 2. Heuristic patterns (any env var containing SECRET, TOKEN, KEY, PASSWORD)
            heuristics = ("_SECRET", "_TOKEN", "_PASSWORD", "_PASSWD", "_PRIVATE_KEY", "_AUTH_KEY")
            for k in list(raw_env.keys()):
                upper_k = k.upper()
                if any(h in upper_k for h in heuristics) and upper_k not in scrubbed_keys:
                    del raw_env[k]
                    scrubbed_keys.append(k)

        # 3. Inject non-interactive batch environment settings
        for k, v in self.process_config.base_env.items():
            raw_env[k] = v

        return raw_env, sorted(scrubbed_keys)

