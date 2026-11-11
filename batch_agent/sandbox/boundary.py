"""
Filesystem boundary validation and sensitive path protection for the sandbox.
Protects against symlink escapes, path traversal attacks, and credential leakage.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from batch_agent.config import SandboxConfig, SandboxProfile


class BoundaryValidator:
    """Enforces strict filesystem read and write boundaries."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    def is_path_sensitive(self, path: str) -> bool:
        """Check if target path matches blacklisted credential or secret patterns."""
        if not path:
            return False

        expanded = os.path.expanduser(path)
        canonical = os.path.realpath(expanded)
        normalized = os.path.abspath(expanded)

        candidates = [path, expanded, canonical, normalized]

        for cand in candidates:
            for pattern in self.config.sensitive_patterns:
                if re.search(pattern, cand, re.IGNORECASE):
                    return True
        return False

    def validate_read_path(self, target_path: str, cwd: str = "") -> Tuple[bool, str]:
        """Validate whether a file can be read under current sandbox policy."""
        if not self.config.enabled:
            return True, ""

        if not target_path or not target_path.strip():
            return False, "Target path is empty"

        # Resolve path against cwd if relative
        base_dir = cwd if cwd else self.config.workspace_root
        if not os.path.isabs(target_path):
            abs_path = os.path.abspath(os.path.join(base_dir, target_path))
        else:
            abs_path = os.path.abspath(os.path.expanduser(target_path))

        # Realpath resolution to thwart symlink escape
        canonical_path = os.path.realpath(abs_path)

        if self.is_path_sensitive(canonical_path) or self.is_path_sensitive(abs_path):
            return False, f"Sandbox security alert: Reading sensitive credential path '{abs_path}' is strictly blocked."

        # In STRICT profile, only paths within workspace or designated read dirs are allowed
        if self.config.profile == SandboxProfile.STRICT:
            allowed_prefixes = [self.config.workspace_root, self.config.scratch_dir] + self.config.allowed_read_dirs
            inside_allowed = any(
                canonical_path == p or canonical_path.startswith(p + os.sep)
                for p in allowed_prefixes
            )
            if not inside_allowed:
                return False, f"Sandbox security alert: In STRICT profile, reads outside workspace '{canonical_path}' are forbidden."

        return True, ""

    def validate_write_path(self, target_path: str, cwd: str = "") -> Tuple[bool, str]:
        """Validate whether a file can be created, edited, or deleted under sandbox policy."""
        if not self.config.enabled:
            return True, ""

        if self.config.profile == SandboxProfile.READ_ONLY:
            return False, "Sandbox violation: Filesystem is mounted in READ_ONLY profile; all write operations are disabled."

        if not target_path or not target_path.strip():
            return False, "Target write path is empty"

        base_dir = cwd if cwd else self.config.workspace_root
        if not os.path.isabs(target_path):
            abs_path = os.path.abspath(os.path.join(base_dir, target_path))
        else:
            abs_path = os.path.abspath(os.path.expanduser(target_path))

        # Check existing realpath or parent realpath for new files
        check_path = abs_path
        if os.path.exists(check_path):
            canonical_path = os.path.realpath(check_path)
        else:
            parent = os.path.dirname(check_path)
            while parent and not os.path.exists(parent):
                parent = os.path.dirname(parent)
            canonical_parent = os.path.realpath(parent) if parent else ""
            canonical_path = os.path.join(canonical_parent, os.path.basename(check_path))

        # Block sensitive targets
        if self.is_path_sensitive(canonical_path) or self.is_path_sensitive(abs_path):
            return False, f"Sandbox violation: Target path '{abs_path}' matches sensitive credentials blacklist."

        # Block writes to critical operating system directories
        system_roots = [
            "/etc", "/usr", "/bin", "/sbin", "/System", "/Library",
            "/boot", "/sys", "/proc", "/dev", "/root"
        ]
        for sys_dir in system_roots:
            if canonical_path == sys_dir or canonical_path.startswith(sys_dir + os.sep):
                return False, f"Sandbox violation: Target path '{abs_path}' resides within protected system hierarchy '{sys_dir}'."

        # Block user shell profiles
        home = os.path.realpath(os.path.expanduser("~"))
        user_startup = [
            os.path.join(home, f) for f in (".bashrc", ".zshrc", ".bash_profile", ".profile", ".zprofile", ".login")
        ]
        if canonical_path in user_startup:
            return False, f"Sandbox violation: Target path '{abs_path}' is a user shell initialization file."

        # Ensure write path is strictly inside allowed write directories
        allowed_dirs = [os.path.realpath(p) for p in self.config.allowed_write_dirs]
        is_inside_allowed = any(
            canonical_path == allowed or canonical_path.startswith(allowed + os.sep)
            for allowed in allowed_dirs
        )

        if not is_inside_allowed:
            return False, (
                f"Sandbox violation: Target write path '{abs_path}' (canonical: '{canonical_path}') "
                f"is outside permitted sandbox boundaries ({', '.join(allowed_dirs)})."
            )

        return True, ""

