"""
Sandbox Manager orchestrating filesystem boundaries, command guarding, and environment scrubbing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from batch_agent.config import ProcessConfig, SandboxConfig, SandboxProfile
from batch_agent.sandbox.boundary import BoundaryValidator
from batch_agent.sandbox.command_guard import CommandGuard
from batch_agent.sandbox.env_sanitizer import EnvironmentSanitizer


class SandboxManager:
    """Central policy coordinator for sandbox security."""

    def __init__(
        self,
        sandbox_config: SandboxConfig | None = None,
        process_config: ProcessConfig | None = None,
    ) -> None:
        self.config = sandbox_config or SandboxConfig()
        self.process_config = process_config or ProcessConfig()

        self.boundary = BoundaryValidator(self.config)
        self.command_guard = CommandGuard(self.config)
        self.sanitizer = EnvironmentSanitizer(self.config, self.process_config)

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    @property
    def profile(self) -> SandboxProfile:
        return self.config.profile

    def set_profile(self, profile: SandboxProfile | str) -> None:
        if isinstance(profile, str):
            self.config.profile = SandboxProfile(profile)
        else:
            self.config.profile = profile

    def validate_read(self, path: str, cwd: str = "") -> Tuple[bool, str]:
        return self.boundary.validate_read_path(path, cwd)

    def validate_write(self, path: str, cwd: str = "") -> Tuple[bool, str]:
        return self.boundary.validate_write_path(path, cwd)

    def validate_command(self, command: str) -> Tuple[bool, str]:
        return self.command_guard.validate_command(command)

    def sanitize_environment(
        self, base_env: Dict[str, str] | None = None
    ) -> Tuple[Dict[str, str], List[str]]:
        return self.sanitizer.sanitize(base_env)

    def check_tool_access(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        cwd: str = "",
    ) -> Tuple[bool, str]:
        """Verify permissions for an arbitrary tool call before execution."""
        if not self.config.enabled:
            return True, ""

        name = tool_name.lower()
        if name in ("file_read", "read_file", "view_file", "read"):
            path = tool_input.get("path") or tool_input.get("file_path") or ""
            return self.validate_read(path, cwd)

        elif name in ("file_write", "write_file", "file_edit", "edit_file", "patch_file"):
            path = tool_input.get("path") or tool_input.get("file_path") or ""
            return self.validate_write(path, cwd)

        elif name in ("file_list", "list_dir", "list_files"):
            path = tool_input.get("path") or tool_input.get("dir_path") or "."
            return self.validate_read(path, cwd)

        elif name in ("bash", "shell", "run_command", "execute"):
            cmd = tool_input.get("command") or tool_input.get("cmd") or ""
            return self.validate_command(cmd)

        return True, ""

