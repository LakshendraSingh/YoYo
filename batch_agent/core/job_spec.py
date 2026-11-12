"""
Batch job specification parser supporting CLI arguments, JSON manifests, and YAML manifests.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from batch_agent.config import (
    AuditConfig,
    BatchAgentConfig,
    LLMProviderType,
    ProcessConfig,
    SandboxConfig,
    SandboxProfile,
)


class BatchJobSpec:
    """Parses and materializes batch job specifications."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BatchAgentConfig:
        goal = data.get("goal", "")
        if not goal:
            raise ValueError("Job specification must include a 'goal' field.")

        job_id = data.get("job_id", "")
        if not job_id:
            job_id = (
                os.environ.get("PBS_JOBID")
                or os.environ.get("SLURM_JOB_ID")
                or os.environ.get("LSB_JOBID")
                or ""
            )
            if job_id and "." in job_id:
                job_id = job_id.split(".")[0]

        associated_files = data.get("associated_files", [])
        raw_ws = data.get("workspace_root") or os.environ.get("PBS_O_WORKDIR") or os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd()
        workspace_root = os.path.realpath(os.path.abspath(raw_ws))

        # Sandbox
        sb_data = data.get("sandbox", {})
        profile_str = sb_data.get("profile", "workspace")
        sandbox_cfg = SandboxConfig(
            enabled=sb_data.get("enabled", True),
            profile=SandboxProfile(profile_str),
            workspace_root=workspace_root,
            allowed_write_dirs=sb_data.get("allowed_write_dirs", []),
            allowed_read_dirs=sb_data.get("allowed_read_dirs", []),
            sensitive_patterns=sb_data.get("sensitive_patterns", []),
            blocked_commands=sb_data.get("blocked_commands", []),
        )

        # Process
        proc_data = data.get("process", {})
        process_cfg = ProcessConfig(
            default_timeout_seconds=proc_data.get("default_timeout_seconds", 60),
            max_output_bytes=proc_data.get("max_output_bytes", 1024 * 1024),
            enforce_non_interactive_stdin=proc_data.get("enforce_non_interactive_stdin", True),
        )

        # Audit
        audit_data = data.get("audit", {})
        audit_cfg = AuditConfig(
            log_file_path=audit_data.get("log_file_path", ""),
            flush_immediately=audit_data.get("flush_immediately", True),
            record_active_processes_snapshot=audit_data.get("record_active_processes_snapshot", True),
        )

        # Provider
        model_weights = data.get("model_weights_path") or data.get("model_weights")
        provider_str = data.get("provider", "local_weights" if model_weights else "heuristic")
        try:
            provider = LLMProviderType(provider_str)
        except ValueError:
            provider = LLMProviderType.LOCAL_WEIGHTS if model_weights else LLMProviderType.HEURISTIC

        return BatchAgentConfig(
            job_id=job_id,
            goal=goal,
            associated_files=associated_files,
            workspace_root=workspace_root,
            max_steps=data.get("max_steps", 30),
            job_timeout_seconds=data.get("job_timeout_seconds", 1800),
            goal_supremacy=data.get("goal_supremacy", True),
            max_token_budget=data.get("max_token_budget", 150000),
            max_consecutive_failed_actions=data.get("max_consecutive_failed_actions", 3),
            enable_recovery_engine=data.get("enable_recovery_engine", True),
            allow_dummy_stubs=data.get("allow_dummy_stubs", False),
            provider=provider,
            model_name=data.get("model_name", "heuristic-batch-v1"),
            model_weights_path=model_weights,
            device=data.get("device", "auto"),
            n_gpu_layers=data.get("n_gpu_layers", data.get("gpu_layers", -1)),
            context_window=data.get("context_window", 4096),
            api_key=data.get("api_key") or os.environ.get("GEMINI_API_KEY"),
            sandbox=sandbox_cfg,
            process=process_cfg,
            audit=audit_cfg,
        )

    @classmethod
    def from_json_file(cls, file_path: str) -> BatchAgentConfig:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def ingest_files(cls, config: BatchAgentConfig) -> List[str]:
        """Verify and ingest associated input files into the sandbox workspace."""
        ingested: List[str] = []
        for file_path in config.associated_files:
            abs_source = os.path.realpath(os.path.abspath(file_path))
            if not os.path.exists(abs_source):
                continue

            # If already inside workspace, keep reference
            if abs_source.startswith(config.workspace_root + os.sep) or abs_source == config.workspace_root:
                rel = os.path.relpath(abs_source, config.workspace_root)
                ingested.append(rel)
            else:
                # Copy file into workspace
                dest = os.path.join(config.workspace_root, os.path.basename(abs_source))
                if abs_source != dest:
                    shutil.copy2(abs_source, dest)
                ingested.append(os.path.basename(dest))
        return ingested

