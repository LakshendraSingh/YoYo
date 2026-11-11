"""
Main autonomous agent orchestration engine designed for non-interactive batch OS environments.
Executes goals end-to-end, enforces sandboxing, manages processes, and produces comprehensive .txt audit trails.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from batch_agent.audit.audit_logger import AuditLogger
from batch_agent.config import BatchAgentConfig
from batch_agent.core.job_spec import BatchJobSpec
from batch_agent.core.state import BatchJobState
from batch_agent.intelligence.planner import ExecutionPlan, StepStatus
from batch_agent.intelligence.provider import BaseProvider, create_provider
from batch_agent.process.cleanup import BatchProcessCleaner
from batch_agent.process.runner import ProcessRunner
from batch_agent.process.tracker import ProcessTracker
from batch_agent.sandbox.sandbox_manager import SandboxManager
from batch_agent.tools.base import ToolContext
from batch_agent.tools.registry import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)


class BatchEngine:
    """The central orchestrator for autonomous non-interactive batch jobs."""

    def __init__(
        self,
        config: BatchAgentConfig,
        registry: Optional[ToolRegistry] = None,
        provider: Optional[BaseProvider] = None,
    ) -> None:
        self.config = config
        self.registry = registry or create_default_registry()

        # Audit Logger
        self.audit_logger = AuditLogger(
            config=self.config.audit,
            workspace_root=self.config.workspace_root,
            job_id=self.config.job_id,
        )

        # Sandbox & Execution
        self.sandbox = SandboxManager(self.config.sandbox, self.config.process)
        self.tracker = ProcessTracker()
        self.runner = ProcessRunner(
            sandbox=self.sandbox,
            tracker=self.tracker,
            audit_logger=self.audit_logger,
            config=self.config.process,
        )

        # Process Preemption Cleaner
        self.cleaner = BatchProcessCleaner(self.tracker, self.audit_logger)
        self.cleaner.install()

        # Intelligence Provider
        self.provider = provider or create_provider(
            provider_type=self.config.provider,
            api_key=self.config.api_key,
            model=self.config.model_name,
            model_weights_path=self.config.model_weights_path,
            device=self.config.device,
            n_gpu_layers=self.config.n_gpu_layers,
            context_window=self.config.context_window,
        )

        # State & Plan (Check for resumption)
        self.checkpoint_path = (
            self.config.checkpoint_file
            or os.path.join(self.config.workspace_root, ".batch_checkpoint.json")
        )
        self._resumed = False

        if self.config.resume and os.path.exists(self.checkpoint_path):
            loaded = BatchJobState.load_checkpoint(self.checkpoint_path)
            if loaded:
                self.state = loaded
                self.plan = loaded.plan
                self._resumed = True
                # Ensure explicitly configured goal maintains priority over checkpoint if provided
                if self.config.goal:
                    self.plan.goal = self.config.goal
                    self.state.goal = self.config.goal

        if not self._resumed:
            self.plan = ExecutionPlan(goal=self.config.goal)
            self.state = BatchJobState(
                job_id=self.audit_logger.job_id,
                goal=self.config.goal,
                workspace_root=self.config.workspace_root,
                plan=self.plan,
            )

    def run(self) -> int:
        """
        Execute the batch job from start to finish without human interaction.
        Returns integer exit status code (0 for success, non-zero for failure).
        """
        start_time = time.time()

        try:
            # 1. Ingest associated files & log start
            ingested_files = BatchJobSpec.ingest_files(self.config)
            self.audit_logger.log_job_start(
                goal=self.config.goal,
                associated_files=ingested_files,
                sandbox_profile=self.sandbox.profile.value,
                max_steps=self.config.max_steps,
                provider=self.config.provider.value,
            )

            # 1b. Technical Document & Research Paper Ingestion
            from batch_agent.documents.analyzer import DocumentAnalyzer
            self.ingested_documents = []
            doc_exts = (".pdf", ".tex", ".latex", ".rst")
            doc_keys = ("srs", "paper", "spec", "requirement", "prd", "rfc")

            for f in ingested_files:
                f_lower = f.lower()
                ext = os.path.splitext(f_lower)[1]
                if ext in doc_exts or any(k in f_lower for k in doc_keys):
                    abs_f = os.path.join(self.config.workspace_root, f)
                    if os.path.exists(abs_f) and os.path.isfile(abs_f):
                        try:
                            digest = DocumentAnalyzer.analyze_file(abs_f)
                            self.ingested_documents.append(digest)
                            from batch_agent.audit.events import AuditEvent, AuditEventType
                            self.audit_logger.record_event(
                                AuditEvent(
                                    event_type=AuditEventType.DOCUMENT_INGESTION,
                                    title=f"Technical Specification Ingested: {digest.title}",
                                    description=(
                                        f"Analyzed {digest.doc_type.value} from '{f}'. "
                                        f"Extracted {len(digest.requirements)} requirements and "
                                        f"{len(digest.algorithms)} algorithms."
                                    ),
                                    metadata={
                                        "file": f,
                                        "doc_type": digest.doc_type.value,
                                        "title": digest.title,
                                        "requirements_count": len(digest.requirements),
                                        "algorithms_count": len(digest.algorithms),
                                        "parameters": digest.key_parameters,
                                    },
                                )
                            )
                        except Exception as e:
                            logger.warning("Could not ingest document %s: %s", f, e)

            # Check if this run is resumed from a prior checkpoint
            if self._resumed:
                from batch_agent.audit.events import AuditEvent, AuditEventType
                completed_count = len([s for s in self.plan.steps if s.status == StepStatus.COMPLETED])
                self.audit_logger.record_event(
                    AuditEvent(
                        event_type=AuditEventType.CHECKPOINT,
                        title="Batch Job Resumed from Checkpoint",
                        description=(
                            f"Resumed execution from step {self.state.current_step_number} "
                            f"with {completed_count}/{len(self.plan.steps)} subgoals already completed."
                        ),
                        metadata={
                            "checkpoint_file": self.checkpoint_path,
                            "resumed_step": self.state.current_step_number,
                            "completed_subgoals_count": completed_count,
                        },
                    )
                )
                print(f"[YOYO] Resuming job from checkpoint at step {self.state.current_step_number} ({completed_count} subgoals completed).")

            # 2. Offline / Air-Gapped Environment & plan.txt Handling
            if self.config.no_internet or self.config.plan_only or self.config.plan_file:
                from batch_agent.intelligence.dependency_planner import DependencyPlanner
                from batch_agent.tools.env_explorer import EnvironmentProbe

                # If user passed an existing plan.txt from a previous run:
                if self.config.plan_file and os.path.exists(self.config.plan_file):
                    loaded_plan = DependencyPlanner.load_plan_file(self.config.plan_file)
                    installed_whls = DependencyPlanner.install_local_wheels_if_available(self.config.workspace_root)
                    inv = EnvironmentProbe.probe(self.config.workspace_root, self.config.associated_files)

                    self.plan = loaded_plan.to_execution_plan()
                    # Goal Priority: If user explicitly provided a goal in config, ensure it overrides/reinforces the plan's goal!
                    if self.config.goal:
                        self.plan.goal = self.config.goal
                    elif loaded_plan.goal:
                        self.config.goal = loaded_plan.goal
                        self.plan.goal = loaded_plan.goal

                    self.state.goal = self.plan.goal
                    self.state.plan = self.plan

                    from batch_agent.audit.events import AuditEvent, AuditEventType
                    self.audit_logger.record_event(
                        AuditEvent(
                            event_type=AuditEventType.CHECKPOINT,
                            title="Offline Plan Ingested from plan.txt",
                            description=f"Loaded plan with {len(self.plan.steps)} subgoals. Installed wheels: {installed_whls or '[None]'}",
                            metadata={
                                "plan_file": self.config.plan_file,
                                "installed_wheels": installed_whls,
                                "available_dependencies_count": len(inv.available_packages),
                                "missing_dependencies": inv.missing_packages,
                            },
                        )
                    )
                else:
                    # Initial run without plan.txt: survey environment and check for missing dependencies
                    plan_path, inventory, has_missing = DependencyPlanner.generate_plan_file(
                        goal=self.config.goal,
                        workspace_root=self.config.workspace_root,
                        associated_files=ingested_files,
                        output_filename="plan.txt",
                    )

                    from batch_agent.audit.events import AuditEvent, AuditEventType
                    self.audit_logger.record_event(
                        AuditEvent(
                            event_type=AuditEventType.CHECKPOINT,
                            title="Offline Plan & Dependency Manifest Generated",
                            description=(
                                f"Surveyed environment. Available dependencies: {len(inventory.available_packages)}, "
                                f"Missing: {len(inventory.missing_packages)}. Saved manifest to {plan_path}."
                            ),
                            metadata={
                                "plan_file": plan_path,
                                "missing_dependencies": inventory.missing_packages,
                                "available_dependencies": inventory.available_packages,
                                "available_binaries": list(inventory.system_binaries.keys()),
                            },
                        )
                    )

                    # If missing dependencies are found or user requested --generate-plan, notify user and pause for dependencies
                    if has_missing or self.config.plan_only:
                        missing_str = ", ".join(inventory.missing_packages) if inventory.missing_packages else "[None]"
                        print(
                            f"\n[YOYO-OFFLINE] Air-gapped / offline environment active (no internet).\n"
                            f"[YOYO-OFFLINE] Generated execution plan and dependency manifest: {plan_path}\n"
                            f"[YOYO-OFFLINE] Available packages in sandbox: {len(inventory.available_packages)}\n"
                            f"[YOYO-OFFLINE] Missing dependencies required: {missing_str}\n"
                            f"[YOYO-OFFLINE] ACTION: Pre-install dependencies or place .whl packages into './wheels/'\n"
                            f"[YOYO-OFFLINE] and resubmit next run with: yoyo --plan plan.txt --no-internet\n"
                        )
                        self.state.is_success = True
                        self.state.exit_code = 0
                        self.state.last_observation = f"Generated offline plan at {plan_path}. Missing dependencies: {missing_str}"
                        self.state.artifacts_produced.append("plan.txt")
                        return 0

            # 3. Main Autonomous Step Loop
            observation = f"Batch environment initialized. Ingested files: {ingested_files}"
            if self.ingested_documents:
                doc_titles = ", ".join(d.title for d in self.ingested_documents)
                observation += f". Ingested {len(self.ingested_documents)} technical specifications/papers: {doc_titles}"
            last_error: Optional[str] = None
            consecutive_failures = 0
            last_action_sig = ""

            cfg_max_steps = self.config.max_steps if self.config.max_steps is not None else 30
            cfg_timeout = self.config.job_timeout_seconds if self.config.job_timeout_seconds is not None else 1800
            cfg_token_budget = self.config.max_token_budget if self.config.max_token_budget is not None else 150000
            cfg_circuit_limit = self.config.max_consecutive_failed_actions if self.config.max_consecutive_failed_actions is not None else 3

            while (cfg_max_steps <= 0) or (self.state.current_step_number < cfg_max_steps):
                self.state.current_step_number += 1
                step_idx = self.state.current_step_number

                # Wall-clock timeout check (vital for batch scheduling boundaries)
                if cfg_timeout > 0 and self.state.elapsed_seconds >= cfg_timeout:
                    timeout_msg = f"Batch job wall-clock timeout exceeded ({cfg_timeout}s)."
                    self.audit_logger.log_error = timeout_msg  # record in audit
                    self.state.error_message = timeout_msg
                    self.state.exit_code = 124
                    break

                # 3. Provider generates Chain of Thought and Decision
                decision = self.provider.decide_next_step(
                    config=self.config,
                    plan=self.plan,
                    step_number=step_idx,
                    observation=observation,
                    last_error=last_error,
                )

                # Token Budget Guard (Runaway Loop Defense)
                step_tokens = (
                    decision.tokens_used
                    if getattr(decision, "tokens_used", None)
                    else max(50, (len(observation) + len(self.config.goal) + 400) // 4)
                )
                self.state.cumulative_tokens_used += step_tokens
                if cfg_token_budget > 0 and self.state.cumulative_tokens_used >= cfg_token_budget:
                    budget_msg = (
                        f"Runaway Loop Guard: Token budget limit reached "
                        f"({self.state.cumulative_tokens_used} >= {cfg_token_budget} tokens). "
                        f"Halting execution to prevent excessive cluster expenses."
                    )
                    from batch_agent.audit.events import AuditEvent, AuditEventType
                    self.audit_logger.record_event(
                        AuditEvent(
                            event_type=AuditEventType.CIRCUIT_BREAKER,
                            step_number=step_idx,
                            title="Runaway Loop Guard: Token Budget Tripped",
                            description=budget_msg,
                            status="TIMED_OUT",
                            tokens_used=self.state.cumulative_tokens_used,
                            metadata={
                                "token_budget": self.config.max_token_budget,
                                "tokens_used": self.state.cumulative_tokens_used,
                            },
                        )
                    )
                    self.state.error_message = budget_msg
                    self.state.exit_code = 1
                    break

                # Log Chain of Thought directly to .txt audit file
                self.audit_logger.log_chain_of_thought(
                    step=step_idx,
                    phase=decision.thought.phase,
                    reasoning=decision.thought.format_text(),
                    plan_summary=self.plan.format_summary(),
                    decision=decision.thought.decision,
                )

                # 4. Handle Terminal Completion Decision
                if decision.is_finish:
                    self.state.is_success = True
                    self.state.exit_code = 0
                    self.state.last_observation = decision.finish_summary
                    break

                # 5. Handle Tool Invocation
                if decision.tool_name:
                    tool_context = ToolContext(
                        workspace_root=self.config.workspace_root,
                        cwd=self.config.workspace_root,
                        sandbox=self.sandbox,
                        tracker=self.tracker,
                        runner=self.runner,
                        audit_logger=self.audit_logger,
                        step_number=step_idx,
                        plan=self.plan,
                        engine=self,
                    )

                    tool_res = self.registry.execute(
                        name=decision.tool_name,
                        params=decision.tool_params,
                        context=tool_context,
                    )

                    current_sig = f"{decision.tool_name}:{decision.tool_params}"

                    if tool_res.success:
                        observation = tool_res.output
                        last_error = None
                        consecutive_failures = 0
                        last_action_sig = ""
                        if self.plan.current_step and self.plan.current_step.status == StepStatus.IN_PROGRESS:
                            self.plan.current_step.mark_completed(tool_res.output[:200])
                            self.plan.advance()
                    else:
                        observation = tool_res.output or ""
                        last_error = tool_res.error or "Tool execution failed"

                        if current_sig == last_action_sig:
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 1
                        last_action_sig = current_sig

                        # Infinite Retry Circuit Breaker (Runaway Loop Defense)
                        if cfg_circuit_limit > 0 and consecutive_failures >= cfg_circuit_limit:
                            loop_msg = (
                                f"Runaway Loop Guard: Action '{decision.tool_name}' failed "
                                f"{consecutive_failures} consecutive times with identical inputs. "
                                f"Tripping circuit breaker to prevent infinite retry cycle."
                            )
                            from batch_agent.audit.events import AuditEvent, AuditEventType
                            self.audit_logger.record_event(
                                AuditEvent(
                                    event_type=AuditEventType.CIRCUIT_BREAKER,
                                    step_number=step_idx,
                                    title="Runaway Loop Guard: Infinite Retry Circuit Breaker Tripped",
                                    description=loop_msg,
                                    status="FAILED",
                                    tokens_used=self.state.cumulative_tokens_used,
                                    metadata={
                                        "failed_action": decision.tool_name,
                                        "consecutive_failures": consecutive_failures,
                                        "goal_supremacy": self.config.goal_supremacy,
                                    },
                                )
                            )
                            if self.config.goal_supremacy:
                                # Pivot! Force skip stuck step so agent can adapt and continue
                                if self.plan.current_step:
                                    self.plan.skip_step(
                                        self.plan.current_step.id,
                                        reason="Tripped infinite retry circuit breaker",
                                    )
                                    self.plan.advance()
                                consecutive_failures = 0
                                last_action_sig = ""
                                last_error = f"Circuit breaker tripped on repeatedly failing action: '{decision.tool_name}'. Strategy pivoted."
                            else:
                                self.state.error_message = loop_msg
                                self.state.exit_code = 1
                                break

                    # Keep on-disk plan.txt synchronized if present in workspace
                    plan_file_path = os.path.join(self.config.workspace_root, "plan.txt")
                    if os.path.exists(plan_file_path):
                        try:
                            from batch_agent.intelligence.dependency_planner import DependencyPlanner
                            DependencyPlanner.update_plan_file(
                                plan_path=plan_file_path,
                                plan=self.plan,
                                mutation_reason=f"Step {step_idx} executed: tool {decision.tool_name}",
                            )
                        except Exception:
                            pass

                # Save checkpoint state after every step
                self.state.save_checkpoint(self.checkpoint_path)

                # 6. Periodic Checkpoints
                if step_idx % 3 == 0:
                    completed_names = [s.title for s in self.plan.steps if s.status == StepStatus.COMPLETED]
                    remaining_names = [s.title for s in self.plan.steps if s.status != StepStatus.COMPLETED]
                    self.audit_logger.log_checkpoint(
                        step=step_idx,
                        completed_steps=completed_names,
                        remaining_steps=remaining_names,
                        active_processes=self.tracker.get_active_snapshots(),
                    )
                    self.state.save_checkpoint()

            # Determine final status if loop ended on step count
            if not self.state.is_success and self.state.exit_code == 0:
                if self.plan.steps and (self.plan.all_completed or self.plan.is_finished):
                    self.state.is_success = True
                    self.state.exit_code = 0
                else:
                    self.state.is_success = False
                    self.state.exit_code = 1
                    self.state.error_message = f"Exhausted maximum step budget ({self.config.max_steps} steps)."

        except Exception as e:
            logger.exception("Uncaught exception in BatchEngine: %s", e)
            self.state.is_success = False
            self.state.exit_code = 1
            self.state.error_message = f"Unhandled exception: {e}"

        finally:
            # Reclaim / terminate all surviving child processes
            self.tracker.terminate_all()
            self.state.save_checkpoint(self.checkpoint_path)

            # Sync final plan status to plan.txt if it exists in workspace
            plan_file_path = os.path.join(self.config.workspace_root, "plan.txt")
            if os.path.exists(plan_file_path):
                try:
                    from batch_agent.intelligence.dependency_planner import DependencyPlanner
                    DependencyPlanner.update_plan_file(
                        plan_path=plan_file_path,
                        plan=self.plan,
                        mutation_reason="Final plan synchronization at job termination",
                    )
                except Exception:
                    pass

            # Record final summary and close audit file
            total_duration = time.time() - start_time
            self.state.end_time = time.time()

            # Discover generated artifacts
            artifacts = self._discover_artifacts()
            summary_text = (
                self.state.last_observation
                if self.state.is_success
                else (self.state.error_message or "Batch task encountered an error.")
            )

            self.audit_logger.log_job_completion(
                success=self.state.is_success,
                total_steps=self.state.current_step_number,
                total_runtime_seconds=total_duration,
                final_summary=summary_text,
                artifacts_produced=artifacts,
                active_processes=self.tracker.get_active_snapshots(),
            )
            self.audit_logger.close()

        return self.state.exit_code

    def _discover_artifacts(self) -> List[str]:
        """Detect new artifacts created in the workspace root."""
        artifacts: List[str] = []
        try:
            for entry in os.listdir(self.config.workspace_root):
                if entry.endswith((".md", ".txt", ".json", ".csv", ".log", ".py", ".png")):
                    if entry != os.path.basename(self.audit_logger.log_path):
                        artifacts.append(entry)
        except Exception:
            pass
        return sorted(artifacts)

