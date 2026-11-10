"""
Configuration module for batch agent execution, sandbox boundaries, and audit logging.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SandboxProfile(str, Enum):
    """Sandbox security boundary levels."""
    STRICT = "strict"              # Filesystem strictly locked to workspace; commands heavily filtered
    WORKSPACE = "workspace"        # Standard mode: read safe host bins, write strictly to workspace & scratch
    READ_ONLY = "read_only"        # Zero write operations permitted anywhere


class LLMProviderType(str, Enum):
    """Supported intelligence / LLM backends."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LOCAL_WEIGHTS = "local_weights"  # Direct execution against local weights (GGUF, Safetensors, Transformers)
    HEURISTIC = "heuristic"        # Fully offline autonomous decision engine (zero API key needed)


@dataclass
class SandboxConfig:
    """Security boundaries and enforcement policies."""
    enabled: bool = True
    profile: SandboxProfile = SandboxProfile.WORKSPACE
    workspace_root: str = ""
    scratch_dir: str = ""
    allowed_write_dirs: List[str] = field(default_factory=list)
    allowed_read_dirs: List[str] = field(default_factory=list)
    sensitive_patterns: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=list)
    scrub_environment: bool = True
    scrubbed_env_vars: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.workspace_root:
            self.workspace_root = os.getcwd()
        self.workspace_root = os.path.realpath(os.path.abspath(self.workspace_root))

        if not self.scratch_dir:
            self.scratch_dir = os.path.join(self.workspace_root, ".batch_scratch")
        self.scratch_dir = os.path.realpath(os.path.abspath(self.scratch_dir))

        if not self.allowed_write_dirs:
            self.allowed_write_dirs = [
                self.workspace_root,
                self.scratch_dir,
                os.path.realpath(os.path.abspath("/tmp")),
            ]
        else:
            self.allowed_write_dirs = [os.path.realpath(os.path.abspath(p)) for p in self.allowed_write_dirs]

        if not self.sensitive_patterns:
            self.sensitive_patterns = [
                r"\.ssh($|/)",
                r"\.aws($|/)",
                r"\.gnupg($|/)",
                r"\.netrc($|/)",
                r"\.bash_history($|/)",
                r"\.zsh_history($|/)",
                r"id_rsa($|\.)",
                r"id_ed25519($|\.)",
                r"\.pem$",
                r"\.key$",
                r"keychain($|/)",
                r"credentials\.json$",
                r"(^|/)\.env($|\..*)",
                r"id_ecdsa($|\.)",
                r"authorized_keys",
                r"known_hosts",
            ]

        if not self.blocked_commands:
            self.blocked_commands = [
                r"\bsudo\b",
                r"\bsu\s+",
                r"\bchmod\s+(-R\s+)?777\b",
                r"\bchown\b",
                r"\bmkfs\b",
                r"\bfdisk\b",
                r"\bdd\s+if=",
                r">+\s*/dev/sd",
                r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
                r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f*|-rf|-fr)\s+(/\s*$|/\*|~\s*$|~\*)",
                r"\bshutdown\b",
                r"\breboot\b",
                r"\binit\s+[06]\b",
                r"\bcurl\b.*\|\s*(ba)?sh\b",
                r"\bwget\b.*\|\s*(ba)?sh\b",
                r"\bwall\b",
                r"\buserdel\b",
                r"\bgroupdel\b",
            ]

        if not self.scrubbed_env_vars:
            self.scrubbed_env_vars = [
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "GITHUB_TOKEN",
                "GITLAB_TOKEN",
                "SSH_AUTH_SOCK",
                "SSH_AGENT_PID",
                "SLURM_JWT",
            ]


@dataclass
class ProcessConfig:
    """Subprocess and system execution constraints."""
    default_timeout_seconds: int = 60
    max_output_bytes: int = 1024 * 1024  # 1 MB stdout/stderr limit
    kill_grace_period_seconds: float = 3.0
    enforce_non_interactive_stdin: bool = True
    base_env: Dict[str, str] = field(default_factory=lambda: {
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "CI": "true",
        "TERM": "dumb",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    })


@dataclass
class AuditConfig:
    """Audit trail configuration for writing the .txt and .csv logs."""
    log_file_path: str = ""
    csv_log_path: str = ""
    enable_csv_log: bool = True
    flush_immediately: bool = True
    record_command_outputs: bool = True
    max_output_lines_in_log: int = 500
    record_active_processes_snapshot: bool = True
    timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%fZ"

    def resolve_path(self, workspace_root: str) -> str:
        if not self.log_file_path:
            return os.path.join(workspace_root, "batch_agent_audit.txt")
        if os.path.isabs(self.log_file_path):
            return self.log_file_path
        return os.path.join(workspace_root, self.log_file_path)

    def resolve_csv_path(self, workspace_root: str) -> str:
        if self.csv_log_path:
            if os.path.isabs(self.csv_log_path):
                return self.csv_log_path
            return os.path.join(workspace_root, self.csv_log_path)
        txt_path = self.resolve_path(workspace_root)
        if txt_path.endswith(".txt"):
            return txt_path[:-4] + ".csv"
        return txt_path + ".csv"


@dataclass
class BatchAgentConfig:
    """Complete batch agent configuration specification."""
    job_id: str = ""
    goal: str = ""
    associated_files: List[str] = field(default_factory=list)
    workspace_root: str = ""
    max_steps: int = 30
    max_retries_per_step: int = 3
    goal_supremacy: bool = True      # Toggle: When True, agent can dynamically alter/mutate plan to achieve goal; when False, plan is strictly followed
    max_token_budget: int = 150000   # Hard token budget limit to protect cluster costs
    max_consecutive_failed_actions: int = 3  # Circuit breaker threshold to prevent runaway loops
    enable_recovery_engine: bool = True      # Autonomous command failure detection & self-healing
    job_timeout_seconds: int = 1800  # 30 minutes wall-clock timeout
    resume: bool = False             # Resume from checkpoint if available
    checkpoint_file: Optional[str] = None  # Explicit path to checkpoint JSON file
    no_internet: bool = False        # Offline / air-gapped environment toggle (generate/use plan.txt)
    plan_file: Optional[str] = None  # Path to existing plan.txt or target output plan.txt
    plan_only: bool = False          # Generate plan.txt and dependency inventory then exit
    provider: LLMProviderType = LLMProviderType.HEURISTIC
    model_name: str = "heuristic-batch-v1"
    model_weights_path: Optional[str] = None
    device: str = "auto"
    n_gpu_layers: int = -1
    context_window: int = 4096
    api_key: Optional[str] = None
    temperature: float = 0.2
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def __post_init__(self) -> None:
        if not self.workspace_root:
            self.workspace_root = os.getcwd()
        self.workspace_root = os.path.realpath(os.path.abspath(self.workspace_root))
        self.sandbox.workspace_root = self.workspace_root
        self.sandbox.__post_init__()

