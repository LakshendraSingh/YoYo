"""
Autonomous Command Failure Detection, Diagnosis, and Self-Healing Recovery Engine.
Enables YoYo to detect and recover from failed shell commands autonomously
without human intervention or crashing the batch job.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from batch_agent.audit.events import AuditEvent, AuditEventType
from batch_agent.process.runner import ProcessResult

logger = logging.getLogger(__name__)


class FailureCategory(str, Enum):
    COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    FILE_OR_PATH_NOT_FOUND = "FILE_OR_PATH_NOT_FOUND"
    SYNTAX_OR_CODE_ERROR = "SYNTAX_OR_CODE_ERROR"
    TIMEOUT_EXPIRED = "TIMEOUT_EXPIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    ASSERTION_OR_TEST_FAILURE = "ASSERTION_OR_TEST_FAILURE"
    GENERAL_ERROR = "GENERAL_ERROR"


@dataclass
class RecoveryAttempt:
    """Detailed record of an autonomous command recovery intervention."""
    original_command: str
    failure_category: FailureCategory
    diagnosis: str
    strategy_applied: str
    repaired_command: Optional[str]
    success: bool
    recovery_output: str = ""
    repaired_result: Optional[ProcessResult] = None


class CommandRecoveryEngine:
    """Diagnoses failed shell subprocesses and executes automated self-healing workarounds."""

    @classmethod
    def diagnose_failure(cls, command: str, result: ProcessResult) -> Tuple[FailureCategory, str]:
        """Classify root cause from exit codes, stdout, and stderr."""
        stderr = result.stderr or ""
        stdout = result.stdout or ""
        combined = f"{stdout}\n{stderr}"

        # 1. Timeout
        if result.timed_out or result.exit_code == 124:
            return FailureCategory.TIMEOUT_EXPIRED, f"Command exceeded execution wall-time limit ({result.duration_seconds:.2f}s)."

        # 2. Command not found (Exit code 127)
        if result.exit_code == 127 or "command not found" in combined.lower() or "not recognized as an internal" in combined.lower():
            cmd_binary = command.strip().split()[0] if command.strip() else "unknown"
            return FailureCategory.COMMAND_NOT_FOUND, f"Executable binary '{cmd_binary}' was not found in system PATH."

        # 3. Missing dependency or module
        if "ModuleNotFoundError:" in combined or "ImportError:" in combined or "No module named" in combined:
            m = re.search(r"No module named\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", combined)
            mod_name = m.group(1) if m else "unknown"
            return FailureCategory.MISSING_DEPENDENCY, f"Python runtime missing required module: '{mod_name}'."

        # 4. File or path not found
        if "no such file or directory" in combined.lower() or "filenotfounderror:" in combined.lower() or "can't open file" in combined.lower() or "cannot open" in combined.lower():
            m = (
                re.search(r"can't open file ['\"]([^'\"]+)['\"]", combined, re.IGNORECASE)
                or re.search(r"No such file or directory:\s*['\"]([^'\"]+)['\"]", combined, re.IGNORECASE)
                or re.search(r"FileNotFoundError:.*?['\"]([^'\"]+)['\"]", combined, re.IGNORECASE)
                or re.search(r"['\"]?([^\r\n'\"]+\.[a-zA-Z0-9_\-]+)['\"]?:\s*(?:\[Errno 2\]\s*)?No such file", combined, re.IGNORECASE)
            )
            path_str = m.group(1) if m else "target file"
            return FailureCategory.FILE_OR_PATH_NOT_FOUND, f"Target path does not exist in workspace: '{path_str}'."

        # 5. Syntax / Indentation error
        if "SyntaxError:" in combined or "IndentationError:" in combined or "TabError:" in combined:
            m = re.search(r'File "([^"]+)", line (\d+)', combined)
            loc = f" in {m.group(1)}:L{m.group(2)}" if m else ""
            return FailureCategory.SYNTAX_OR_CODE_ERROR, f"Python code syntax or indentation defect detected{loc}."

        # 6. Permission denied
        if "Permission denied" in combined or result.exit_code == 126:
            return FailureCategory.PERMISSION_DENIED, "Subprocess encountered filesystem permission or access restriction."

        # 7. Unit test failure
        if "AssertionError" in combined or "FAILED (failures=" in combined or "FAIL:" in combined:
            return FailureCategory.ASSERTION_OR_TEST_FAILURE, "Automated test assertions failed during execution."

        return FailureCategory.GENERAL_ERROR, f"Command exited with non-zero status code: {result.exit_code}."

    @classmethod
    def attempt_recovery(
        cls,
        command: str,
        result: ProcessResult,
        context: Any,
        plan: Optional[Any] = None,
    ) -> RecoveryAttempt:
        """
        Formulate and execute an autonomous self-healing recovery intervention.
        Returns a RecoveryAttempt capturing the outcome and any repaired process result.
        """
        category, diagnosis = cls.diagnose_failure(command, result)
        workspace_root = getattr(context, "workspace_root", os.getcwd())
        runner = getattr(context, "runner", None)
        audit_logger = getattr(context, "audit_logger", None)
        step_number = getattr(context, "step_number", None)

        logger.info("Attempting autonomous recovery for failed command [%s]: %s", category.value, diagnosis)

        # Do not intercept test assertion failures; they provide legitimate testing feedback to the LLM agent.
        if category == FailureCategory.ASSERTION_OR_TEST_FAILURE:
            return RecoveryAttempt(
                original_command=command,
                failure_category=category,
                diagnosis=diagnosis,
                strategy_applied="None (test assertion failure is diagnostic feedback for agent)",
                repaired_command=None,
                success=False,
            )

        # -------------------------------------------------------------
        # Strategy A: Command Not Found (Substitute Alternate Binaries)
        # -------------------------------------------------------------
        if category == FailureCategory.COMMAND_NOT_FOUND and runner:
            repaired_cmd = None
            tokens = command.strip().split()
            binary = tokens[0] if tokens else ""

            if binary in ("python", "python3.11", "python3.10", "python3.9"):
                repaired_cmd = f"{sys.executable} " + " ".join(tokens[1:])
            elif binary == "pip":
                repaired_cmd = f"{sys.executable} -m pip " + " ".join(tokens[1:])
            elif binary == "pytest":
                target_args = tokens[1:]
                has_file = any(t.endswith(".py") for t in target_args)
                if has_file or not target_args:
                    repaired_cmd = f"{sys.executable} -m unittest " + " ".join(target_args)
                else:
                    repaired_cmd = f"{sys.executable} -m unittest discover " + " ".join(target_args)

            if repaired_cmd:
                new_res = runner.run(repaired_cmd, cwd=context.cwd, step_number=step_number)
                success = (new_res.exit_code == 0)
                attempt = RecoveryAttempt(
                    original_command=command,
                    failure_category=category,
                    diagnosis=diagnosis,
                    strategy_applied=f"Substituted missing binary '{binary}' with active environment executable: '{sys.executable}'",
                    repaired_command=repaired_cmd,
                    success=success,
                    recovery_output=new_res.stdout or new_res.stderr,
                    repaired_result=new_res,
                )
                cls._log_recovery(attempt, audit_logger, step_number)
                return attempt

        # -------------------------------------------------------------
        # Strategy B: File or Path Not Found (Workspace Discovery & Relink)
        # -------------------------------------------------------------
        if category == FailureCategory.FILE_OR_PATH_NOT_FOUND and runner:
            combined = f"{result.stdout}\n{result.stderr}"
            m = (
                re.search(r"can't open file ['\"]([^'\"]+)['\"]", combined, re.IGNORECASE)
                or re.search(r"No such file or directory:\s*['\"]([^'\"]+)['\"]", combined, re.IGNORECASE)
                or re.search(r"FileNotFoundError:.*?['\"]([^'\"]+)['\"]", combined, re.IGNORECASE)
                or re.search(r"['\"]?([^\r\n'\"]+\.[a-zA-Z0-9_\-]+)['\"]?:\s*(?:\[Errno 2\]\s*)?No such file", combined, re.IGNORECASE)
            )
            target_path = m.group(1).strip() if m else None

            if target_path:
                basename = os.path.basename(target_path)
                # Search workspace for actual location
                found_rel: Optional[str] = None
                for root, _, files in os.walk(workspace_root):
                    if basename in files:
                        full = os.path.join(root, basename)
                        found_rel = os.path.relpath(full, context.cwd)
                        break

                if found_rel:
                    if target_path in command:
                        repaired_cmd = command.replace(target_path, found_rel)
                    elif basename in command:
                        repaired_cmd = command.replace(basename, found_rel)
                    else:
                        repaired_cmd = f"{command} {found_rel}"

                    new_res = runner.run(repaired_cmd, cwd=context.cwd, step_number=step_number)
                    success = (new_res.exit_code == 0)
                    attempt = RecoveryAttempt(
                        original_command=command,
                        failure_category=category,
                        diagnosis=diagnosis,
                        strategy_applied=f"Located shifted file in workspace at '{found_rel}' and corrected command path.",
                        repaired_command=repaired_cmd,
                        success=success,
                        recovery_output=new_res.stdout or new_res.stderr,
                        repaired_result=new_res,
                    )
                    cls._log_recovery(attempt, audit_logger, step_number)
                    return attempt

        # -------------------------------------------------------------
        # Strategy C: Missing Dependency (Wheel Search & Runner Translation)
        # -------------------------------------------------------------
        FORBIDDEN_STUB_MODULES = {
            "pytest", "pytest_cov", "coverage", "nose", "unittest",
            "tox", "mypy", "ruff", "flake8", "black", "pylint"
        }

        if category == FailureCategory.MISSING_DEPENDENCY and runner:
            combined = f"{result.stdout}\n{result.stderr}"
            m = re.search(r"No module named\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", combined)
            mod_name = m.group(1) if m else None

            if mod_name:
                # 1. If missing test runner like pytest, translate to Python standard library unittest!
                if mod_name in ("pytest", "nose") or ("pytest" in command):
                    repaired_cmd = re.sub(r"\bpython3?\s+-m\s+pytest\b", f"{sys.executable} -m unittest", command)
                    if repaired_cmd == command and "pytest" in command:
                        repaired_cmd = re.sub(r"\bpytest\b", f"{sys.executable} -m unittest", command)
                    if repaired_cmd != command:
                        new_res = runner.run(repaired_cmd, cwd=context.cwd, step_number=step_number)
                        success = (new_res.exit_code == 0)
                        attempt = RecoveryAttempt(
                            original_command=command,
                            failure_category=category,
                            diagnosis=f"Missing third-party test framework '{mod_name}'.",
                            strategy_applied="Translated command to Python standard library unittest to genuinely execute test assertions.",
                            repaired_command=repaired_cmd,
                            success=success,
                            recovery_output=new_res.stdout or new_res.stderr,
                            repaired_result=new_res,
                        )
                        cls._log_recovery(attempt, audit_logger, step_number)
                        return attempt

                # 2. Check if matching wheel exists in workspace or wheels/
                from batch_agent.intelligence.dependency_planner import DependencyPlanner
                installed = DependencyPlanner.install_local_wheels_if_available(workspace_root)
                if installed:
                    new_res = runner.run(command, cwd=context.cwd, step_number=step_number)
                    if new_res.exit_code == 0:
                        attempt = RecoveryAttempt(
                            original_command=command,
                            failure_category=category,
                            diagnosis=diagnosis,
                            strategy_applied=f"Autonomously installed local wheel packages ({', '.join(installed)}) and retried.",
                            repaired_command=command,
                            success=True,
                            recovery_output=new_res.stdout,
                            repaired_result=new_res,
                        )
                        cls._log_recovery(attempt, audit_logger, step_number)
                        return attempt

                # 3. Only synthesize safe fallback shim module if explicitly opted in (disabled by default in production)
                engine_ref = getattr(context, "engine", None)
                allow_stubs = getattr(getattr(engine_ref, "config", None), "allow_dummy_stubs", False) if engine_ref else False
                if allow_stubs and mod_name not in FORBIDDEN_STUB_MODULES:
                    shim_path = os.path.join(context.cwd, f"{mod_name}.py")
                    if not os.path.exists(shim_path):
                        shim_code = (
                            f'"""Autonomous fallback stub for missing module: {mod_name}"""\n'
                            f'import sys\n'
                            f'class _FallbackStub:\n'
                            f'    def __getattr__(self, name):\n'
                            f'        return lambda *args, **kwargs: None\n'
                            f'sys.modules["{mod_name}"] = _FallbackStub()\n'
                        )
                        try:
                            with open(shim_path, "w", encoding="utf-8") as f:
                                f.write(shim_code)
                            new_res = runner.run(command, cwd=context.cwd, step_number=step_number)
                            success = (new_res.exit_code == 0)
                            attempt = RecoveryAttempt(
                                original_command=command,
                                failure_category=category,
                                diagnosis=diagnosis,
                                strategy_applied=f"Synthesized pure standard-library fallback stub for '{mod_name}' to continue batch execution.",
                                repaired_command=command,
                                success=success,
                                recovery_output=new_res.stdout or new_res.stderr,
                                repaired_result=new_res,
                            )
                            cls._log_recovery(attempt, audit_logger, step_number)
                            return attempt
                        except Exception:
                            pass

        # -------------------------------------------------------------
        # Strategy D: Timeout Exceeded (Adjust Non-Interactive Batch Flags)
        # -------------------------------------------------------------
        if category == FailureCategory.TIMEOUT_EXPIRED and runner:
            repaired_cmd = None
            if "unittest" in command and "-v" in command:
                repaired_cmd = command.replace("-v", "-q")
            elif "python" in command and "--batch" not in command:
                repaired_cmd = command + " --batch"

            if repaired_cmd and repaired_cmd != command:
                new_res = runner.run(repaired_cmd, cwd=context.cwd, step_number=step_number)
                success = (new_res.exit_code == 0)
                attempt = RecoveryAttempt(
                    original_command=command,
                    failure_category=category,
                    diagnosis=diagnosis,
                    strategy_applied="Injected non-interactive batch flags to reduce console I/O deadlock risk.",
                    repaired_command=repaired_cmd,
                    success=success,
                    recovery_output=new_res.stdout or new_res.stderr,
                    repaired_result=new_res,
                )
                cls._log_recovery(attempt, audit_logger, step_number)
                return attempt

        # -------------------------------------------------------------
        # Strategy E: LLM-Driven Autonomous Self-Healing
        # -------------------------------------------------------------
        engine = getattr(context, "engine", None)
        provider = getattr(engine, "provider", None) if engine else None
        goal = getattr(engine.config, "goal", "") if engine and hasattr(engine, "config") else ""

        if runner and provider and hasattr(provider, "repair_failed_command"):
            logger.info("Engaging LLM provider for autonomous command repair...")
            llm_repair = provider.repair_failed_command(
                goal=goal,
                command=command,
                exit_code=result.exit_code,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
            if llm_repair and llm_repair.get("repaired_command"):
                repaired_cmd = llm_repair["repaired_command"].strip()
                if repaired_cmd != command:
                    logger.info("Executing LLM-repaired command: %s", repaired_cmd)
                    new_res = runner.run(repaired_cmd, cwd=context.cwd, step_number=step_number)
                    success = (new_res.exit_code == 0)
                    attempt = RecoveryAttempt(
                        original_command=command,
                        failure_category=category,
                        diagnosis=llm_repair.get("diagnosis", diagnosis),
                        strategy_applied=f"LLM Self-Healing ({provider.__class__.__name__}): {llm_repair.get('strategy', 'AI command repair')}",
                        repaired_command=repaired_cmd,
                        success=success,
                        recovery_output=new_res.stdout or new_res.stderr,
                        repaired_result=new_res,
                    )
                    cls._log_recovery(attempt, audit_logger, step_number)
                    return attempt

        # Default fallback attempt: recorded as unresolved but handled gracefully
        attempt = RecoveryAttempt(
            original_command=command,
            failure_category=category,
            diagnosis=diagnosis,
            strategy_applied="No automated direct patch viable; recorded diagnostic context for planning engine self-correction.",
            repaired_command=None,
            success=False,
            recovery_output=result.stderr or result.stdout,
        )
        cls._log_recovery(attempt, audit_logger, step_number)
        return attempt

    @classmethod
    def _log_recovery(
        cls,
        attempt: RecoveryAttempt,
        audit_logger: Optional[Any],
        step_number: Optional[int],
    ) -> None:
        """Record recovery diagnosis and action in both .txt and .csv audit streams."""
        if not audit_logger:
            return

        status_str = "RECOVERED" if attempt.success else "FAILED_RECOVERY"
        event = AuditEvent(
            event_type=AuditEventType.RECOVERY_ACTION,
            step_number=step_number,
            title=f"Command Self-Healing: [{attempt.failure_category.value}] -> {status_str}",
            description=(
                f"Autonomous diagnosis: {attempt.diagnosis}\n"
                f"Self-healing strategy: {attempt.strategy_applied}\n"
                f"Result: {status_str}"
            ),
            command=attempt.repaired_command or attempt.original_command,
            exit_code=attempt.repaired_result.exit_code if attempt.repaired_result else 1,
            duration_seconds=attempt.repaired_result.duration_seconds if attempt.repaired_result else 0.0,
            status=status_str,
            recovery_details=attempt.strategy_applied,
            metadata={
                "failure_category": attempt.failure_category.value,
                "original_command": attempt.original_command,
                "repaired_command": attempt.repaired_command,
                "strategy": attempt.strategy_applied,
                "recovered": attempt.success,
            },
        )
        audit_logger.record_event(event)
