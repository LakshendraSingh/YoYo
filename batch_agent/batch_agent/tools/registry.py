"""
Central Tool Registry managing tool discovery, schema generation, and sandbox-checked invocation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from batch_agent.tools.ast_tool import CodeASTInspectTool
from batch_agent.tools.base import BaseTool, ToolContext, ToolResult
from batch_agent.tools.env_explorer import EnvironmentDiscoveryTool
from batch_agent.tools.doc_tool import DocumentIngestTool
from batch_agent.tools.execution import CommandExecutionTool
from batch_agent.tools.filesystem import (
    FileEditTool,
    FileListTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
)
from batch_agent.tools.plan_tool import UpdatePlanTool
from batch_agent.tools.process_tool import ProcessInspectTool
from batch_agent.tools.verification import VerifyCodeTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry holding all available tools for autonomous batch execution."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name.lower()] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name.lower())

    def list_tool_names(self) -> List[str]:
        return sorted(list(self._tools.keys()))

    def get_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return schemas

    def execute(
        self,
        name: str,
        params: Dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a tool with sandbox policy gatekeeping and audit logging."""
        tool = self.get(name)
        if not tool:
            err = f"Unknown tool: '{name}'. Available: {', '.join(self.list_tool_names())}"
            if context.audit_logger:
                context.audit_logger.log_tool_invocation(
                    step=context.step_number or 0,
                    tool_name=name,
                    parameters=params,
                    sandbox_passed=False,
                    result_summary="",
                    error=err,
                )
            return ToolResult(success=False, error=err)

        # 1. Gatecheck with sandbox manager
        allowed, reason = context.sandbox.check_tool_access(name, params, context.cwd)
        if not allowed:
            if context.audit_logger:
                context.audit_logger.log_security_alert(
                    step=context.step_number,
                    violation=reason,
                    attempted_target=str(params),
                    policy_rule=f"ToolGuard:{name}",
                )
                context.audit_logger.log_tool_invocation(
                    step=context.step_number or 0,
                    tool_name=name,
                    parameters=params,
                    sandbox_passed=False,
                    result_summary="",
                    error=f"Sandbox blocked: {reason}",
                )
            return ToolResult(success=False, error=f"Sandbox violation: {reason}")

        # 2. Execute tool
        try:
            result = tool.run(params, context)
            summary = result.output[:300] if result.success else (result.error or "")[:300]

            if context.audit_logger:
                context.audit_logger.log_tool_invocation(
                    step=context.step_number or 0,
                    tool_name=name,
                    parameters=params,
                    sandbox_passed=True,
                    result_summary=summary,
                    error=result.error,
                )

            return result
        except Exception as e:
            err_msg = f"Unexpected tool execution error: {e}"
            logger.exception(err_msg)
            if context.audit_logger:
                context.audit_logger.log_tool_invocation(
                    step=context.step_number or 0,
                    tool_name=name,
                    parameters=params,
                    sandbox_passed=True,
                    result_summary="",
                    error=err_msg,
                )
            return ToolResult(success=False, error=err_msg)


def create_default_registry() -> ToolRegistry:
    """Factory creating registry populated with standard batch tools."""
    registry = ToolRegistry()
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(FileListTool())
    registry.register(FileSearchTool())
    registry.register(CommandExecutionTool())
    registry.register(ProcessInspectTool())
    registry.register(VerifyCodeTool())
    registry.register(CodeASTInspectTool())
    registry.register(EnvironmentDiscoveryTool())
    registry.register(UpdatePlanTool())
    registry.register(DocumentIngestTool())
    return registry

