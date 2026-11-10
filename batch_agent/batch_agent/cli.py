"""
Command-line interface for running YoYo in non-interactive batch environments.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from batch_agent.config import (
    AuditConfig,
    BatchAgentConfig,
    LLMProviderType,
    ProcessConfig,
    SandboxConfig,
    SandboxProfile,
)
from batch_agent.core.engine import BatchEngine
from batch_agent.core.job_spec import BatchJobSpec


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yoyo",
        description=(
            "YoYo: Autonomous sandboxed agentic AI system for non-interactive batch OS environments.\n\n"
            "DISCLAIMER & LIMITATION OF LIABILITY:\n"
            "YoYo operates autonomously and can make mistakes or modify workspace code/data. "
            "Always execute jobs on dedicated copies and maintain independent backups of critical data.\n"
            "The author(s) and contributors accept no liability for any loss, damage, or data corruption arising from its use."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--goal", "-g",
        type=str,
        default="",
        help="Primary goal description to complete autonomously (top priority in each run)",
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="",
        help="Path to batch job JSON manifest file",
    )
    parser.add_argument(
        "--plan", "-p",
        type=str,
        default="",
        help="Path to plan.txt generated from previous offline run",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume execution from existing checkpoint",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to specific .batch_checkpoint.json file to resume from",
    )

    parser.add_argument(
        "--files", "-f",
        nargs="*",
        default=[],
        help="List of associated files or inputs relevant to the goal",
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default="",
        help="Target workspace root directory (default: current directory)",
    )
    parser.add_argument(
        "--log-file", "-l",
        type=str,
        default="",
        help="Output .txt audit log path (default: batch_agent_audit.txt in workspace)",
    )
    parser.add_argument(
        "--csv-log", "--csv-file",
        type=str,
        default="",
        help="Output machine-parseable .csv audit log path (default: batch_agent_audit.csv in workspace)",
    )
    parser.add_argument(
        "--goal-supremacy",
        dest="goal_supremacy",
        action="store_true",
        default=True,
        help="Enable goal supremacy mode: agent has full authority to alter/replan steps to accomplish goal (default)",
    )
    parser.add_argument(
        "--strict-plan", "--no-goal-supremacy",
        dest="goal_supremacy",
        action="store_false",
        help="Disable goal supremacy: agent strictly adheres to plan steps without mutating them",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=150000,
        help="Hard ceiling on cumulative LLM token consumption to prevent runaway loop costs (default: 150000; 0 to disable)",
    )
    parser.add_argument(
        "--no-token-budget",
        action="store_true",
        help="Disable token budget guard (allow unlimited LLM tokens)",
    )
    parser.add_argument(
        "--no-step-limit",
        action="store_true",
        help="Disable step limit guard (allow unlimited reasoning steps)",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable wall-clock timeout guard (allow unlimited execution time)",
    )
    parser.add_argument(
        "--subprocess-timeout",
        type=int,
        default=60,
        help="Per-subprocess execution timeout limit in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-subprocess-timeout",
        action="store_true",
        help="Disable subprocess execution timeout guard",
    )
    parser.add_argument(
        "--circuit-breaker-limit", "--max-retries",
        dest="circuit_breaker_limit",
        type=int,
        default=3,
        help="Consecutive failed identical actions before tripping infinite retry circuit breaker (default: 3)",
    )
    parser.add_argument(
        "--no-circuit-breaker",
        action="store_true",
        help="Disable consecutive failure action circuit breaker",
    )
    parser.add_argument(
        "--recovery-engine",
        dest="enable_recovery",
        action="store_true",
        default=True,
        help="Enable self-healing command recovery engine (default: True)",
    )
    parser.add_argument(
        "--no-recovery",
        dest="enable_recovery",
        action="store_false",
        help="Disable self-healing command recovery engine",
    )
    parser.add_argument(
        "--no-internet", "--offline",
        action="store_true",
        help="Toggle offline/air-gapped mode: explores environment and saves plan.txt if dependencies missing",
    )
    parser.add_argument(
        "--generate-plan",
        action="store_true",
        help="Analyze environment and write plan.txt with dependency manifest then exit",
    )
    parser.add_argument(
        "--sandbox-profile",
        choices=["strict", "workspace", "read_only"],
        default="workspace",
        help="Sandbox isolation profile boundary level (default: workspace)",
    )
    parser.add_argument(
        "--provider",
        choices=["heuristic", "gemini", "openai", "anthropic", "ollama", "local_weights"],
        default="heuristic",
        help="Intelligence provider engine (default: heuristic for offline batch clusters)",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="",
        help="Path to local model weights (GGUF file, Safetensors file, or HuggingFace model directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device for local model weights (default: auto)",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of model layers to offload to GPU (-1 for all layers, default: -1)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=4096,
        help="Context window size in tokens for local weights (default: 4096)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Maximum autonomous reasoning steps allowed (default: 25)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Total job wall-clock timeout in seconds (default: 1800)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not (args.goal or args.manifest or args.plan or args.resume or args.checkpoint):
        parser.error("At least one of --goal (-g), --manifest (-m), --plan (-p), or --resume (-r) must be specified.")

    effective_max_steps = 0 if args.no_step_limit else (args.max_steps if args.max_steps is not None else 25)
    effective_timeout = 0 if args.no_timeout else (args.timeout if args.timeout is not None else 1800)
    effective_token_budget = 0 if args.no_token_budget else (args.token_budget if args.token_budget is not None else 150000)
    effective_sub_timeout = 0 if args.no_subprocess_timeout else (args.subprocess_timeout if args.subprocess_timeout is not None else 60)
    effective_circuit_breaker = 0 if args.no_circuit_breaker else (args.circuit_breaker_limit if args.circuit_breaker_limit is not None else 3)

    try:
        if args.manifest:
            manifest_path = os.path.abspath(args.manifest)
            if not os.path.exists(manifest_path):
                sys.stderr.write(f"Error: Manifest file '{manifest_path}' does not exist.\n")
                return 1
            config = BatchJobSpec.from_json_file(manifest_path)
            # Goal priority: Allow CLI --goal to override or refine manifest goal
            if args.goal:
                config.goal = args.goal
            # Override with CLI workspace or log if provided
            if args.workspace:
                config.workspace_root = os.path.abspath(args.workspace)
            if args.log_file:
                config.audit.log_file_path = args.log_file
            if args.csv_log:
                config.audit.csv_log_path = args.csv_log
            config.goal_supremacy = args.goal_supremacy
            config.max_token_budget = effective_token_budget
            config.process.default_timeout_seconds = effective_sub_timeout
            config.max_consecutive_failed_actions = effective_circuit_breaker
            config.enable_recovery_engine = args.enable_recovery
            if args.no_step_limit or args.max_steps != 25:
                config.max_steps = effective_max_steps
            if args.no_timeout or args.timeout != 1800:
                config.job_timeout_seconds = effective_timeout
        else:
            raw_ws = args.workspace or os.environ.get("PBS_O_WORKDIR") or os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd()
            ws_root = os.path.realpath(os.path.abspath(raw_ws))
            detected_job_id = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or ""
            if detected_job_id and "." in detected_job_id:
                detected_job_id = detected_job_id.split(".")[0]

            # If user provided --plan, parse goal and steps from plan.txt
            target_goal = args.goal or ""
            if args.plan:
                from batch_agent.intelligence.dependency_planner import DependencyPlanner
                abs_plan = os.path.abspath(args.plan) if not os.path.isabs(args.plan) else args.plan
                if os.path.exists(abs_plan):
                    loaded_plan = DependencyPlanner.load_plan_file(abs_plan)
                    if not target_goal:
                        target_goal = loaded_plan.goal or f"Execute plan from {os.path.basename(abs_plan)}"

            # If user requested --resume, verify checkpoint existence
            is_resume = args.resume or bool(args.checkpoint)
            checkpoint_target = args.checkpoint or os.path.join(ws_root, ".batch_checkpoint.json")
            if is_resume and not target_goal:
                if os.path.exists(checkpoint_target):
                    from batch_agent.core.state import BatchJobState
                    st = BatchJobState.load_checkpoint(checkpoint_target)
                    if st:
                        target_goal = st.goal
                if not target_goal:
                    target_goal = "Resume execution from checkpoint"

            sb_profile = SandboxProfile(args.sandbox_profile)
            prov_type = LLMProviderType.LOCAL_WEIGHTS if args.model_weights else LLMProviderType(args.provider)

            config = BatchAgentConfig(
                job_id=detected_job_id,
                goal=target_goal,
                associated_files=args.files,
                workspace_root=ws_root,
                max_steps=effective_max_steps,
                job_timeout_seconds=effective_timeout,
                resume=is_resume,
                checkpoint_file=args.checkpoint or None,
                provider=prov_type,
                model_weights_path=args.model_weights or None,
                device=args.device,
                n_gpu_layers=args.gpu_layers,
                context_window=args.context_window,
                no_internet=args.no_internet or bool(args.plan),
                plan_file=args.plan or None,
                plan_only=args.generate_plan,
                goal_supremacy=args.goal_supremacy,
                max_token_budget=effective_token_budget,
                max_consecutive_failed_actions=effective_circuit_breaker,
                enable_recovery_engine=args.enable_recovery,
                sandbox=SandboxConfig(
                    enabled=True,
                    profile=sb_profile,
                    workspace_root=ws_root,
                ),
                process=ProcessConfig(
                    default_timeout_seconds=effective_sub_timeout,
                ),
                audit=AuditConfig(
                    log_file_path=args.log_file,
                    csv_log_path=args.csv_log,
                ),
            )

        # Run engine
        engine = BatchEngine(config)
        exit_code = engine.run()

        # Batch job exit message
        status_text = "SUCCESS" if exit_code == 0 else f"FAILED (Exit code {exit_code})"
        print(f"[YOYO] Job completed: {status_text}")
        print(f"[YOYO] Audit trail recorded: {engine.audit_logger.log_path}")
        return exit_code

    except Exception as e:
        sys.stderr.write(f"[YOYO-ERROR] Fatal exception: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

