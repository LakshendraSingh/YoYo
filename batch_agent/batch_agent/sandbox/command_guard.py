"""
Command execution safety guard for sandbox validation in batch environments.
Blocks dangerous commands, interactive utilities, and sensitive pattern access.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from batch_agent.config import SandboxConfig, SandboxProfile


class CommandGuard:
    """Evaluates shell commands for security hazards and batch-incompatibility."""

    # Commands that require interactive TTY / user inputs and will hang batch jobs
    INTERACTIVE_UTILITIES = [
        r"\b(vi|vim|nano|emacs|pico)\b",
        r"\b(top|htop|atop|glances)\b",
        r"\b(less|more)\b",
        r"\bman\s+",
        r"\bpasswd\b",
        r"\bgdb\b(?!.*--batch)",
        r"\bpython3?\s+-i\b",
        r"\bsh\s+-i\b",
        r"\bbash\s+-i\b",
    ]

    # Safe inspection prefixes for read-only profile
    READ_ONLY_PREFIXES = (
        "ls", "cat", "grep", "find", "head", "tail", "wc", "git status",
        "git log", "git diff", "pwd", "echo", "which", "file", "diff",
        "env", "python3 -c", "pytest", "stat"
    )

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    def validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate if a shell command is safe and compatible with batch execution."""
        if not self.config.enabled:
            return True, ""

        if not command or not command.strip():
            return False, "Command is empty"

        cleaned = command.strip()

        # Check read-only mode restriction
        if self.config.profile == SandboxProfile.READ_ONLY:
            allowed_readonly = any(cleaned.startswith(p) for p in self.READ_ONLY_PREFIXES)
            if not allowed_readonly:
                return False, f"Sandbox security violation (READ_ONLY profile): Modifying or unapproved command '{cleaned[:40]}' is disallowed."

        # Check dangerous command patterns
        for pattern in self.config.blocked_commands:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return False, f"Sandbox security alert: Prohibited dangerous pattern '{pattern}' detected in command."

        # Check sensitive file patterns in command string
        for pattern in self.config.sensitive_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return False, f"Sandbox security alert: Command references sensitive credential pattern '{pattern}'."

        # Check for interactive utilities that will hang batch jobs waiting on stdin
        for pattern in self.INTERACTIVE_UTILITIES:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return False, (
                    f"Batch execution error: Command references interactive utility '{pattern}' which "
                    "requires interactive TTY / stdin. Batch operating systems do not support interactive prompts."
                )

        return True, ""

