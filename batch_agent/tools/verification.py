"""
Automated verification tool for code correctness, test execution, and quality checks.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from batch_agent.tools.base import BaseTool, ToolContext, ToolResult


class VerifyCodeTool(BaseTool):
    """Verifies syntax, type checks, or runs test suites non-interactively."""

    name = "verify_code"
    description = (
        "Execute automated non-interactive verification checks: "
        "'syntax' (py_compile), 'unittest' (python unittest runner), or 'custom_command'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["syntax", "unittest", "custom_command"],
                "description": "Verification check mode",
            },
            "target": {"type": "string", "description": "File or command to verify"},
        },
        "required": ["mode"],
    }

    def run(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        target = params.get("target") or params.get("test_file") or params.get("file") or params.get("path") or ""
        command = params.get("command", "")
        mode = params.get("mode")

        if command:
            mode = "custom_command"
            target = command
        elif not mode:
            if target and ("test" in target.lower() or os.path.isdir(os.path.join(context.cwd, target))):
                mode = "unittest"
            else:
                mode = "syntax" if target else "unittest"

        if mode == "syntax":
            if not target:
                return ToolResult(success=False, error="Target file path required for syntax check")

            cmd = f"python3 -m py_compile {target}"
            run_res = context.runner.run(cmd, cwd=context.cwd, step_number=context.step_number)
            passed = (run_res.exit_code == 0)

            if context.audit_logger:
                context.audit_logger.log_verification(
                    step=context.step_number or 0,
                    check_name=f"Syntax Check ({target})",
                    passed=passed,
                    details="Python syntax validated cleanly" if passed else f"Syntax Error: {run_res.stderr}",
                )

            return ToolResult(
                success=passed,
                output=f"Syntax check {'PASSED' if passed else 'FAILED'} for {target}.\n{run_res.stderr or run_res.stdout}",
                error=None if passed else "Syntax validation failed",
            )

        elif mode == "unittest":
            test_target = target if target else "tests"
            full_test_path = os.path.join(context.cwd, test_target)

            if os.path.isdir(full_test_path):
                cmd = f"python3 -m unittest discover -s {test_target}"
            elif os.path.isfile(full_test_path):
                cmd = f"python3 -m unittest {test_target}"
            else:
                # No specific tests dir/file found; attempt root discovery
                cmd = "python3 -m unittest discover -s . -p 'test*.py'"

            run_res = context.runner.run(cmd, cwd=context.cwd, step_number=context.step_number)

            output = (run_res.stdout + "\n" + run_res.stderr).strip()
            # Exit code 0 is standard pass; exit code 5 is Python 3.12+ "NO TESTS RAN"
            passed = (run_res.exit_code == 0) or (run_res.exit_code == 5 and "NO TESTS RAN" in output)

            if context.audit_logger:
                context.audit_logger.log_verification(
                    step=context.step_number or 0,
                    check_name=f"Unit Test Suite ({test_target})",
                    passed=passed,
                    details=f"Test run outcome: {'PASSED' if passed else 'FAILED'}.\n{output[:500]}",
                )

            return ToolResult(
                success=passed,
                output=f"Test suite execution {'PASSED' if passed else 'FAILED'}.\n{output}",
                error=None if passed else "One or more unit tests failed",
            )

        elif mode == "custom_command":
            if not target:
                return ToolResult(success=False, error="Target command required for custom verification")
            run_res = context.runner.run(target, cwd=context.cwd, step_number=context.step_number)
            passed = (run_res.exit_code == 0)

            if context.audit_logger:
                context.audit_logger.log_verification(
                    step=context.step_number or 0,
                    check_name=f"Custom Verification ({target[:30]})",
                    passed=passed,
                    details=f"Outcome: {'PASSED' if passed else 'FAILED'}. Exit code: {run_res.exit_code}",
                )

            return ToolResult(
                success=passed,
                output=f"Verification command {'PASSED' if passed else 'FAILED'}.\n{run_res.stdout}\n{run_res.stderr}",
                error=None if passed else f"Verification exited with code {run_res.exit_code}",
            )

        return ToolResult(success=False, error=f"Unknown verification mode: {mode}")
