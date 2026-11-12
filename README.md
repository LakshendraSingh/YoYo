# YoYo — Autonomous Sandboxed Agentic AI for Batch Operating Environments

> **A robust, self-contained, sandboxed agentic AI system engineered specifically for non-interactive batch OS environments (HPC Slurm, PBS Pro / OpenPBS / Torque, AWS Batch, GCP Batch, Kubernetes Jobs, CI/CD runners, and headless daemon nodes).**

> [!WARNING]
> **Safety Disclaimer, Data Protection Notice & Limitation of Liability**:
> YoYo is an autonomous agentic AI system designed to operate unattended without human-in-the-loop oversight during batch job runs. **YoYo can make mistakes**, generate flawed code, or alter/overwrite workspace files.
> - **Always use dedicated working copies** of your project repositories and scratch directories rather than executing directly in active production trees.
> - **Always maintain independent, verified backups** of all critical code, datasets, configurations, and models prior to launching batch jobs.
> - **Limitation of Liability**: The software is provided "as is", without warranty of any kind, express or implied. **In no event shall the author(s), contributors, or copyright holders be liable for any claim, damages, data loss, code corruption, system downtime, cluster cost overrun, or other liability**, whether in an action of contract, tort or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software. Users assume all responsibility and risk for the autonomous actions executed by the agent.

---

## Table of Contents
1. [Executive Summary & Problem Context](#1-executive-summary--problem-context)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Capabilities & Engineering Principles](#3-core-capabilities--engineering-principles)
   - [Non-Interactive Batch Guarantees & Subprocess Lifecycle](#1-non-interactive-batch-guarantees--subprocess-lifecycle)
   - [Hardened Multi-Tier Sandboxing & Path Boundary Verification](#2-hardened-multi-tier-sandboxing--path-boundary-verification)
   - [Runaway Loop Risk Defenses (Fully Toggleable)](#3-runaway-loop-risk-defenses-fully-toggleable)
   - [Autonomous Command Failure Diagnosis & Self-Healing Recovery](#4-autonomous-command-failure-diagnosis--self-healing-recovery)
   - [Dual Audit Logging: Human .txt & Machine-Parseable .csv](#5-dual-audit-logging-human-txt--machine-parseable-csv)
   - [Goal Supremacy Mode vs Strict Plan Execution](#6-goal-supremacy-mode-vs-strict-plan-execution)
   - [Offline / Air-Gapped 2-Stage Planning & Execution](#7-offline--air-gapped-2-stage-planning--execution)
   - [Direct Local Model Weights Connection (GGUF & Safetensors)](#8-direct-local-model-weights-connection-gguf--safetensors)
   - [Technical Document, Research Paper & SRS Ingestion](#9-technical-document-research-paper--srs-ingestion)
   - [Crash-Resilient Checkpointing & Resumption](#10-crash-resilient-checkpointing--resumption)
4. [Directory & Codebase Structure](#4-directory--codebase-structure)
5. [Exhaustive CLI Reference & Command Matrix](#5-exhaustive-cli-reference--command-matrix)
6. [Cluster Schedulers & Deployment Guides](#6-cluster-schedulers--deployment-guides)
   - [Slurm HPC Submission](#slurm-hpc-submission)
   - [PBS / PBS Pro / OpenPBS / Torque Submission](#pbs--pbs-pro--openpbs--torque-submission)
   - [Portable Cron, Systemd & CI/CD Runner](#portable-cron-systemd--cicd-runner)
   - [JSON Batch Manifest Execution](#json-batch-manifest-execution)
7. [Audit Trail Sample Outputs](#7-audit-trail-sample-outputs)
   - [Human-Readable .txt Audit Log Sample](#human-readable-txt-audit-log-sample)
   - [Machine-Parseable .csv Audit Log Sample](#machine-parseable-csv-audit-log-sample)
8. [Verification & Comprehensive Test Suite](#8-verification--comprehensive-test-suite)

---

## 1. Executive Summary & Problem Context

Traditional interactive AI coding assistants rely on continuous back-and-forth communication with a human operator (prompting for permissions before running terminal commands, asking clarifying questions, or requesting approval on code diffs). 

However, in **Batch Operating Systems** and cluster schedulers (Slurm, PBS, AWS Batch, headless worker daemons):
- Standard input (`stdin`) is disconnected (`/dev/null`) or closed. Interactive prompts (`input()`, confirmation dialogs, `readline`, password prompts) cause immediate fatal `EOFError` crashes, broken pipes, or deadlocks that exhaust cluster wall-clock allocations.
- Without an interactive human operator to monitor progress, agents face the risk of **Runaway Loops**—wasting thousands of GPU hours or cloud API credits repeatedly attempting broken actions.
- Unmonitored child processes can turn into orphaned zombies or exhaust cluster compute limits.
- Clusters frequently preempt or re-queue jobs without warning, requiring instant, crash-resilient disk persistence.

**YoYo** is an enterprise-grade agent engineered specifically to solve these challenges with an autonomous, self-healing, zero-mandatory-dependency Python architecture.

---

## 2. Architecture Overview

```
                                  +---------------------------------------+
                                  |     BATCH OS SCHEDULER / RUNNER       |
                                  |   (Slurm / PBS / AWS Batch / Cron)    |
                                  +---------------------------------------+
                                                      |
                                    [Goal + Manifest + Associated Files]
                                                      v
+-----------------------------------------------------------------------------------------------------+
|                                                 YOYO                                                |
|                                                                                                     |
|  +--------------------+        +-----------------------+        +--------------------------------+  |
|  | BatchJobSpec       | -----> | BatchEngine           | -----> | ExecutionPlan                  |  |
|  | - Goal Ingestion   |        | - Autonomous Loop     |        | - Dynamic Replanning           |  |
|  | - File Verification|        | - Timeout Guard       |        | - Goal Decomposition           |  |
|  +--------------------+        +-----------------------+        +--------------------------------+  |
|                                            |                                                        |
|                 +--------------------------+--------------------------+                             |
|                 |                                                     |                             |
|                 v                                                     v                             |
|  +--------------------------------+                 +--------------------------------+              |
|  | Intelligence & Decision Engine |                 | Hardened Multi-Tier Sandbox    |              |
|  | - Chain of Thought (CoT) Engine|                 | - Workspace Boundary Validator |              |
|  | - Heuristic Offline Planner    |                 | - Symlink Traversal Protection |              |
|  | - Direct Local Weights (GGUF)  |                 | - Credential Blacklist (.env)  |              |
|  | - Goal Supremacy Controller    |                 | - Command Guard (No sudo/rm-rf)|              |
|  +--------------------------------+                 | - Subprocess Env Sanitizer     |              |
|                 |                                   +--------------------------------+              |
|                 |                                                     |                             |
|                 v                                                     v                             |
|  +-----------------------------------------------------------------------------------------------+  |
|  | Autonomous Tool & Self-Healing Engine                                                         |  |
|  | - CommandRecoveryEngine (Binary substitution, path relinking, wheel discovery, shims)        |  |
|  | - Resilient Multi-Stage Fuzzy Patcher (Exact -> Normalized -> Indent-Shift -> SequenceMatcher)  |  |
|  | - Technical Document & Research Paper Parser (PDF zlib, LaTeX, Markdown, SRS requirements)   |  |
|  | - Non-Interactive Process Runner (DEVNULL stdin, timeout control, process group isolation)    |  |
|  | - Automated Code Verification (Syntax py_compile, Unit Tests, Custom Assertions)              |  |
|  +-----------------------------------------------------------------------------------------------+  |
|                 |                                                                                   |
|                 v                                                                                   |
|  +-----------------------------------------------------------------------------------------------+  |
|  | Active Process Registry & Preemption Handler                                                  |  |
|  | - PID tracking, process duration monitoring, resource tracking                                |  |
|  | - POSIX Signal traps (SIGTERM / SIGINT) for graceful cluster preemption termination           |  |
|  | - Zero-orphan cleanup via atexit and emergency kill handlers                                  |  |
|  +-----------------------------------------------------------------------------------------------+  |
|                 |                                                                                   |
+-----------------|-----------------------------------------------------------------------------------+
                  v
+-----------------------------------------------------------------------------------------------------+
|                              DUAL CRASH-RESILIENT AUDIT SYSTEM                                      |
|  - Human-Readable .txt Log: ASCII banners, CoT reasoning, process tables, step-by-step outcomes      |
|  - Machine-Parseable .csv Log: Strict ISO-8601 UTC timestamps, step IDs, exit codes, tokens used    |
|  - Synchronous fsync to disk on every event (zero data loss under cluster preemption)              |
+-----------------------------------------------------------------------------------------------------+
```

---

## 3. Core Capabilities & Engineering Principles

### 1. Non-Interactive Batch Guarantees & Subprocess Lifecycle
- **Zero Stdin Deadlocks**: All child subprocesses execute with `stdin=subprocess.DEVNULL` and isolated session groups (`start_new_session=True`). Any process requesting input receives immediate EOF instead of hanging.
- **Process Lifecycle & Cleanup**: The `ProcessTracker` registers every spawned PID. On normal exit, error, or OS preemption (`SIGTERM`/`SIGINT`), `BatchProcessCleaner` terminates all child processes cleanly—preventing orphaned processes from lingering on batch nodes.
- **Process Snapshots in Audit Trails**: Active child processes (PID, status, command, elapsed seconds) are captured in every audit log entry.

### 2. Hardened Multi-Tier Sandboxing & Path Boundary Verification
- **Filesystem Confinement**: Strict boundary checks restrict all writes strictly to the designated workspace root and scratch directories. System directories (`/etc`, `/usr`, `/bin`, `/System`, `/boot`) and user shell startup scripts (`.bashrc`, `.zshrc`) are permanently blocked.
- **Symlink Traversal Defense**: Canonical `os.path.realpath` resolution blocks directory escapes through symlink junctions.
- **Credential & Secret Protection**: Strict regex filtering intercepts read and write attempts targeting sensitive files (`~/.ssh`, `~/.aws`, `.env`, `id_rsa`, `keychain`, `*.pem`, `*.key`).
- **Command Safety Guard**: Intercepts destructive shell operations (`sudo`, `su`, `mkfs`, fork bombs `:(){ :|:& };:`, `rm -rf /`, `shutdown`, `reboot`) and interactive terminal commands (`vim`, `nano`, `top`, `less`, `man`).
- **Environment Scrubbing**: Automatically strips API keys and secret tokens (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `SSH_AUTH_SOCK`, `SLURM_JWT`), and injects non-interactive variables (`DEBIAN_FRONTEND=noninteractive`, `PYTHONUNBUFFERED=1`, `CI=true`).

### 3. Runaway Loop Risk Defenses (Fully Toggleable)
Without a human operator to interrupt a stuck agent in a batch queue, YoYo enforces multi-layered defenses against runaway loops. **Every single defense is fully configurable and toggleable** via CLI flags or JSON manifests:

| Defense Layer | Default | How to Tune | How to Toggle Off / Disable | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Hard Step Budget** | `25` steps | `--max-steps <N>` | `--no-step-limit` or `--max-steps 0` | Caps reasoning steps before halting cleanly with exit code 1. |
| **Wall-Clock Job Timeout** | `1800`s (30m) | `--timeout <SEC>` | `--no-timeout` or `--timeout 0` | Prevents exceeding batch scheduler allocations. |
| **Subprocess Execution Timeout** | `60`s | `--subprocess-timeout <SEC>` | `--no-subprocess-timeout` or `--subprocess-timeout 0` | Kills hung shell commands cleanly via process group signal. |
| **Cumulative Token Budget** | `150,000` tokens | `--token-budget <N>` | `--no-token-budget` or `--token-budget 0` | Halts execution if LLM consumption exceeds budget to prevent unexpected cluster/API expenses. |
| **Infinite Retry Circuit Breaker** | `3` identical fails | `--circuit-breaker-limit <N>` | `--no-circuit-breaker` or `--circuit-breaker-limit 0` | Detects when the same failing action is repeated with identical inputs and trips to avoid infinite retry loops. |
| **Command Recovery Engine** | `True` (enabled) | `--recovery-engine` | `--no-recovery` | Self-heals failed shell commands rather than crashing or looping. |
| **Goal Supremacy Toggle** | `True` (enabled) | `--goal-supremacy` | `--strict-plan` or `--no-goal-supremacy` | Under circuit breakers, adapts and mutates the plan to accomplish the goal vs strictly failing. |

*Setting any numerical parameter to `0` or passing its `--no-...` toggle gracefully disables that defense without premature termination.*

### 4. Autonomous Command Failure Diagnosis & Self-Healing Recovery
Instead of crashing the batch job or looping when a command fails, YoYo's [`CommandRecoveryEngine`](batch_agent/process/recovery.py) detects the failure pattern and attempts self-healing:
1. **Automated Root-Cause Classification**:
   - `COMMAND_NOT_FOUND` (Exit 127): Missing executable in `$PATH`.
   - `FILE_OR_PATH_NOT_FOUND`: Shifted relative path or missing input file.
   - `MISSING_DEPENDENCY`: `ModuleNotFoundError` or `ImportError`.
   - `SYNTAX_OR_CODE_ERROR`: Python `SyntaxError` or traceback in workspace code.
   - `ASSERTION_OR_TEST_FAILURE`: Python (`unittest`/`pytest`), C/C++ (`GoogleTest`, `CTest`, `Catch2`, `assert()`), and Go test failures pass straight back to the agent as diagnostic feedback without command hijacking.
   - `TIMEOUT_EXPIRED` (Exit 124): Subprocess hung or took too long.
   - `PERMISSION_DENIED`: Blocked access.
2. **Self-Healing Recovery Interventions**:
   - **Binary Substitution**: If `python`, `pip`, or `pytest` is not found, discovers and substitutes active environment binaries (`sys.executable`, `python3 -m unittest`).
   - **Workspace Path Relinking**: Searches non-ignored workspace subdirectories for shifted filenames (e.g. `calc.py` -> `src/calc.py`), rewrites the command path, and re-executes.
   - **Wheel Search & Opt-In Shims**: Checks `./wheels/` for local packages. Dummy fallback stub creation is disabled by default (`allow_dummy_stubs=False`) and permanently forbidden for test/lint frameworks (`pytest`, `unittest`, `mypy`, `ruff`, etc.).
   - **LLM-Driven Command Repair**: Engages the active LLM provider (`GeminiProvider` / `OpenAIProvider`) to diagnose and synthesize a repaired shell invocation on non-test command failures.
3. **O(1) Scaling & Polyglot Production Features**:
   - **Cross-Step Working Memory**: `GeminiProvider` and `OpenAIProvider` maintain a bounded 8-step rolling execution history (~1,400 tokens max) for multi-file reasoning.
   - **`.gitignore` & Build Pruning**: `file_list`, `file_search`, `file_read`, and `file_edit` automatically prune `.git`, `.venv`/`venv`, `node_modules`, `dist`, `build`, `cmake-build-debug`, `cmake-build-release`, `target`, and `.gitignore` patterns.
   - **Huge Single-File On-Disk Drill-Down**: `file_search` supports `path` + `context_lines`, `ast_inspect` supports `query` symbol filtering, and `file_read` guards against unbounded reads on >1,000-line files.
   - **Python & C/C++ AST Inspection**: `ast_inspect` extracts classes, structs, methods, functions, and imports/includes for both Python (`.py`) and C/C++ (`.c`, `.cpp`, `.cc`, `.h`, `.hpp`) with zero external dependencies.
4. **Audit Trail Logging**:
   - Successfully recovered commands are logged under `[RECOVERY_ACTION]` with status `RECOVERED`, preserving batch job continuity.

### 5. Dual Audit Logging: Human .txt & Machine-Parseable .csv
To satisfy both human operators and automated observability pipelines:
- **Human-Readable .txt Logs (`--log-file`)**: Clear ASCII banners, step headers, indented chain-of-thought blocks, process snapshot tables, and final summaries for cluster operators reviewing runs via `cat` or `less`.
- **Machine-Parseable .csv Logs (`--csv-log`)**: Strict ISO-8601 UTC timestamps, step indices, event types, exit codes, durations, commands, tokens used, and active PIDs formatted for log aggregators (Splunk, Datadog, Prometheus exporters, or Pandas):
  ```csv
  timestamp,step,event_type,title,status,exit_code,duration_seconds,command,tokens_used,active_pids,error_summary,description
  2026-11-10T20:15:30.123456Z,1,JOB_START,Batch Job Ingestion,SUCCESS,0,0.002,,0,0,,Ingested goal and associated files
  2026-11-10T20:15:31.456789Z,2,COMMAND_EXECUTION,Shell Execution,SUCCESS,0,0.124,python3 -m unittest discover,500,0,,Command executed cleanly
  ```
- **Synchronous Persistence**: Both `.txt` and `.csv` files are flushed and fsynced after every single event to survive node preemption.

### 6. Goal Supremacy Mode vs Strict Plan Execution
YoYo supports two operational philosophies depending on cluster policies:
- **Goal Supremacy Mode (`--goal-supremacy`, default)**:
  - The primary goal is the overriding priority across all runs.
  - The plan is a living guide: YoYo has full autonomy to alter the plan (`update_plan`), add subgoals, skip obsolete steps, or completely replan whenever necessary to accomplish the goal.
  - Synchronizes all adaptations to `plan.txt` with mutation rationale.
- **Strict Plan Execution Mode (`--strict-plan` / `--no-goal-supremacy`)**:
  - For audited or regulated cluster pipelines where step sequences must be strictly adhered to without variance.
  - YoYo executes the loaded plan steps strictly in order without altering, adding, or skipping steps.

### 7. Offline / Air-Gapped 2-Stage Planning & Execution
In secure, firewalled batch clusters with **no internet access**, YoYo provides a specialized 2-stage planning and execution workflow:

#### Stage 1: Environment Exploration & Plan Generation
Run with `--no-internet`:
```bash
python3 -m batch_agent.cli \
    --goal "Execute deep learning analytics pipeline on data/samples.csv" \
    --files "pipeline.py" "data/samples.csv" \
    --no-internet \
    --workspace "./my_workspace"
```
1. **Autonomous Environment Probe**: The agent autonomously inspects the sandbox environment, surveying available system binaries (`gcc`, `git`, `make`), installed Python packages, hardware accelerators (CUDA/GPU), and parses all AST code imports across the codebase.
2. **Identification & Difference Matching**: It cross-references needed dependencies against what is **already available** on the host.
3. **`proposal/` Directory Code Drafting**: The agent creates `proposal/` in the workspace and drafts the complete implementation:
   - `proposal/solution.py`: Production-grade executable code implementing the goal with dependency imports, data parsing, and compute routines.
   - `proposal/test_solution.py`: Unit test suite verifying functionality once dependencies are installed.
4. **`plan.txt` Manifest Creation**: Generates a structured `plan.txt` containing:
   - Full environment inventory (binaries, packages, GPU status)
   - List of already available dependencies
   - List of **missing dependencies**
   - **Detailed Package Rationale & Next Actions**: Explains exactly **WHY** each package is needed and **WHAT** the agent plans to do next with it once available
   - **Proposed Code Architecture**: Explicitly links to `proposal/solution.py` and `proposal/test_solution.py`
   - Sequential execution subgoals and actionable user instructions
5. **User Notification**: Outputs instructions to stdout and records the checkpoint in the audit log, then exits cleanly.

#### Stage 2: Ingesting `plan.txt` & Execution in the Next Batch Run
The user or cluster administrator pre-packages the missing wheels or installs the dependencies:
```bash
# Option A: Download binary wheels on an internet-connected machine
mkdir -p ./my_workspace/wheels
pip download -d ./my_workspace/wheels pandas scipy

# Option B: Or install them into the cluster environment/module
pip install pandas scipy
```
Then resubmits the batch job pointing to `plan.txt`:
```bash
python3 -m batch_agent.cli \
    --plan "./my_workspace/plan.txt" \
    --workspace "./my_workspace" \
    --no-internet
```
The agent loads `plan.txt`, automatically installs local wheels from `./wheels/` if provided, verifies all dependencies are satisfied, and executes the proposed programs in `proposal/` end-to-end.

### 8. Direct Local Model Weights Connection (GGUF & Safetensors)
YoYo can run fully offline on compute nodes by connecting directly to local model weights without APIs, network calls, or cloud subscriptions:
- **GGUF Format**: Leverages local CPU or GPU offload (via llama.cpp / metal / CUDA).
- **HuggingFace / Safetensors**: Direct weight loading from local disk paths.
- **CLI Options**: `--model-weights <PATH>`, `--device {auto,cuda,cpu,mps}`, `--gpu-layers <N>`, `--context-window <N>`.

```bash
python3 -m batch_agent.cli \
    --goal "Fix broken test cases in tests/test_calc.py" \
    --model-weights "/shared/models/llama-3-8b-instruct.Q4_K_M.gguf" \
    --gpu-layers 33 \
    --device cuda \
    --workspace "./my_workspace"
```

### 9. Technical Document, Research Paper & SRS Ingestion
YoYo can ingest, semantically analyze, and adhere to technical documentation provided alongside batch jobs:
- **Broad Format Support**:
  - **Research Papers**: PDFs (via 3-tier parsing: installed python libraries, `pdftotext` binary, and native zero-dependency standard-library `zlib` stream decompressor), LaTeX / BibTeX (`.tex`, `.bib`), and Markdown preprints.
  - **SRS / Specifications**: Software Requirements Specifications (`.md`, `.txt`, `.rst`), Product Requirement Documents (PRDs), HTML documentation, and JSON/YAML specs.
- **Semantic Extraction**:
  - Automatically identifies document type (`RESEARCH_PAPER`, `SRS_SPECIFICATION`, `ALGORITHM_GUIDE`, `TECHNICAL_SPEC`).
  - Extracts structured **Functional Requirements** (`REQ-01`, `FR-02`, "The system shall...", "Must/Shall" criteria).
  - Extracts **Algorithmic Specifications & Pseudocode** (`Algorithm 1: ...`, mathematical procedures, input/output schemas).
  - Extracts **Configuration Parameters** (learning rates, batch sizes, thresholds, timeouts).
- **Execution & Code Alignment**:
  - Pre-flight automated document ingestion logs `[DOCUMENT_INGESTION]` directly to the `.txt` audit trail.
  - Automatically enriches LLM system prompts and the offline heuristic engine with the extracted requirements.
  - Injects requirement compliance checks into `proposal/solution.py` and `proposal/test_solution.py` so drafted programs strictly implement the scientific paper or SRS requirements.
  - Dedicated agent tool [`ingest_document`](batch_agent/tools/doc_tool.py) (`mode`: `full`, `requirements`, `algorithms`, `summary`) callable on-demand during runtime.

### 10. Crash-Resilient Checkpointing & Resumption
In batch operating systems, jobs may be interrupted by cluster preemption, wall-clock timeouts, or system restarts. YoYo continuously saves crash-resilient checkpoints (`.batch_checkpoint.json`) to disk using synchronous `fsync`:
- **Resume directly**:
  ```bash
  python3 -m batch_agent.cli --resume --workspace "./my_workspace"
  ```
- **Resume from a specific checkpoint file**:
  ```bash
  python3 -m batch_agent.cli --resume --checkpoint "./my_workspace/.batch_checkpoint.json"
  ```
When resumed, the agent restores completed subgoals, previous file state, and step numbers, logs a `[CHECKPOINT_RESUMED]` event into the audit trail, and continues execution seamlessly without repeating completed work.

---

## 4. Directory & Codebase Structure

```
batch_agent/
├── README.md                      # Comprehensive system documentation & user guide
├── pyproject.toml                 # Standard packaging metadata
├── setup.py                       # Packaging setup script
├── batch_agent/
│   ├── __init__.py                # Package exports
│   ├── cli.py                     # Non-interactive CLI entry point (yoyo)
│   ├── config.py                  # Sandbox, process, audit, and agent configuration
│   ├── documents/
│   │   ├── __init__.py
│   │   ├── models.py              # DocumentDigest, RequirementSpec, AlgorithmSpec
│   │   ├── parser.py              # Multi-tier PDF (native zlib + libraries), LaTeX, Markdown, HTML
│   │   └── analyzer.py            # Requirements, algorithm, and parameter extraction
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── audit_logger.py        # Crash-resilient synchronous .txt & .csv audit logger
│   │   ├── events.py              # Strongly-typed audit event definitions
│   │   └── formatter.py           # Structured ASCII text block formatter
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── boundary.py            # Filesystem path and symlink boundary validator
│   │   ├── command_guard.py       # Dangerous and interactive command safety filter
│   │   ├── env_sanitizer.py       # Secret scrubbing and non-interactive env injector
│   │   └── sandbox_manager.py     # Central policy enforcement coordinator
│   ├── process/
│   │   ├── __init__.py
│   │   ├── tracker.py             # Active process registry and duration monitor
│   │   ├── runner.py              # Non-interactive subprocess runner (DEVNULL stdin)
│   │   ├── recovery.py            # Command failure diagnosis and self-healing recovery engine
│   │   └── cleanup.py             # POSIX signal traps and emergency process killer
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseTool and ToolContext abstractions
│   │   ├── registry.py            # Tool discovery and gated execution
│   │   ├── filesystem.py          # Read, write, list, search, and resilient fuzzy patcher
│   │   ├── execution.py           # Sandboxed command execution tool with self-healing recovery
│   │   ├── process_tool.py        # Process inspector and runaway PID terminator
│   │   ├── verification.py        # Non-interactive syntax, test suite, and check runner
│   │   ├── plan_tool.py           # Dynamic execution plan mutator (update_plan)
│   │   ├── env_explorer.py        # Sandbox inventory & offline dependency planner
│   │   ├── ast_tool.py            # AST symbol inspector and code structure analyzer
│   │   └── doc_tool.py            # Technical document & paper analyzer (ingest_document)
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── provider.py            # LLM provider interface (Gemini, OpenAI, Heuristic, Weights)
│   │   ├── prompt_builder.py      # Non-interactive batch prompt formatter
│   │   ├── planner.py             # Dynamic execution plan and replanning logic
│   │   ├── dependency_planner.py  # 2-stage offline manifest generator & loader
│   │   └── cot_engine.py          # Chain of Thought reasoning generator
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py               # Live job state and checkpoint tracking
│   │   ├── engine.py              # Main autonomous execution loop
│   │   └── job_spec.py            # JSON/CLI job manifest specification parser
│   └── templates/
│       ├── slurm_job.sbatch       # Production Slurm HPC job submission script
│       ├── pbs_job.pbs            # PBS Pro / OpenPBS / Torque submission script
│       ├── run_batch.sh           # Portable headless runner for cron, systemd, and CI/CD
│       └── Dockerfile             # Containerized batch worker image definition
├── examples/
│   ├── job_data_pipeline.json     # Data processing batch job specification
│   ├── job_code_fix.json          # Automated test verification and repair job
│   └── sample_workspace/          # Sample batch input files and test scripts
└── tests/
    ├── __init__.py
    ├── test_sandbox.py            # Sandbox boundary and command guard tests
    ├── test_process_tracker.py    # Process lifecycle and non-interactive stdin tests
    ├── test_audit_logger.py       # .txt and .csv audit trail formatting and flush tests
    ├── test_tools.py              # Filesystem, fuzzy patching, and verification tests
    ├── test_autonomous_engine.py  # End-to-end non-interactive autonomous execution tests
    ├── test_cli.py                # Command-line interface and manifest parsing tests
    ├── test_checkpoint.py         # Crash-resilient checkpointing and resumption tests
    ├── test_offline_planner.py    # 2-stage offline environment exploration & plan.txt tests
    ├── test_local_weights.py      # Direct model weights connection and AST inspection tests
    ├── test_goal_priority_and_replanning.py # Goal supremacy and dynamic replanning tests
    ├── test_runaway_loop_and_recovery.py # Hard limits, circuit breakers, recovery, dual logs
    └── test_document_ingestion.py # Research papers (PDF/LaTeX) and SRS document ingestion
```

---

## 5. Exhaustive CLI Reference & Command Matrix

YoYo is invoked via the `yoyo` entrypoint or `python3 -m batch_agent.cli`:

```bash
usage: yoyo [-h] [--goal GOAL] [--manifest MANIFEST] [--plan PLAN] [--resume]
            [--checkpoint CHECKPOINT] [--files [FILES ...]]
            [--workspace WORKSPACE] [--log-file LOG_FILE] [--csv-log CSV_LOG]
            [--goal-supremacy] [--strict-plan] [--token-budget TOKEN_BUDGET]
            [--no-token-budget] [--no-step-limit] [--no-timeout]
            [--subprocess-timeout SUBPROCESS_TIMEOUT]
            [--no-subprocess-timeout]
            [--circuit-breaker-limit CIRCUIT_BREAKER_LIMIT]
            [--no-circuit-breaker] [--recovery-engine] [--no-recovery]
            [--no-internet] [--generate-plan]
            [--sandbox-profile {strict,workspace,read_only}]
            [--provider {heuristic,gemini,openai,anthropic,ollama,local_weights}]
            [--model-weights MODEL_WEIGHTS] [--device {auto,cuda,cpu,mps}]
            [--gpu-layers GPU_LAYERS] [--context-window CONTEXT_WINDOW]
            [--max-steps MAX_STEPS] [--timeout TIMEOUT]
```

### Options & Parameter Descriptions

| CLI Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--goal`, `-g` | String | `""` | Primary goal description to complete autonomously (top priority in each run). |
| `--manifest`, `-m` | Path | `""` | Path to batch job JSON manifest file. |
| `--plan`, `-p` | Path | `""` | Path to `plan.txt` generated from previous offline run. |
| `--resume`, `-r` | Flag | `False` | Resume execution from existing `.batch_checkpoint.json`. |
| `--checkpoint` | Path | `""` | Path to specific `.batch_checkpoint.json` file to resume from. |
| `--files`, `-f` | List | `[]` | List of associated files, research papers, or inputs relevant to the goal. |
| `--workspace`, `-w` | Path | Current dir | Target workspace root directory. |
| `--log-file`, `-l` | Path | `batch_agent_audit.txt` | Output human-readable `.txt` audit log path. |
| `--csv-log` | Path | `batch_agent_audit.csv` | Output machine-parseable `.csv` audit log path. |
| `--goal-supremacy` | Flag | `True` | **Goal Supremacy Mode**: Agent has full authority to mutate/replan steps to achieve goal. |
| `--strict-plan` | Flag | `False` | **Strict Plan Mode**: Agent strictly follows plan steps in order without mutations. |
| `--token-budget` | Integer | `150000` | Hard ceiling on cumulative LLM tokens before tripping circuit breaker (`0` to disable). |
| `--no-token-budget` | Flag | `False` | Disables token budget guard (allows unlimited LLM tokens). |
| `--max-steps` | Integer | `25` | Maximum autonomous reasoning steps allowed before halting. |
| `--no-step-limit` | Flag | `False` | Disables step limit guard (allows unlimited reasoning steps). |
| `--timeout` | Integer | `1800` | Total job wall-clock timeout in seconds (default: 30 minutes). |
| `--no-timeout` | Flag | `False` | Disables wall-clock timeout guard (allows unlimited execution time). |
| `--subprocess-timeout` | Integer | `60` | Per-subprocess execution timeout limit in seconds. |
| `--no-subprocess-timeout` | Flag | `False` | Disables per-subprocess execution timeout. |
| `--circuit-breaker-limit` | Integer | `3` | Consecutive identical action failures before tripping circuit breaker. |
| `--no-circuit-breaker` | Flag | `False` | Disables consecutive failure action circuit breaker. |
| `--recovery-engine` | Flag | `True` | Enables autonomous command failure diagnosis and self-healing. |
| `--no-recovery` | Flag | `False` | Disables self-healing command recovery engine. |
| `--no-internet`, `--offline` | Flag | `False` | Offline/air-gapped mode: explores environment and saves `plan.txt` if packages missing. |
| `--generate-plan` | Flag | `False` | Analyzes environment, drafts `proposal/`, writes `plan.txt`, and exits. |
| `--sandbox-profile` | Choice | `workspace` | Sandbox boundary profile: `strict`, `workspace`, or `read_only`. |
| `--provider` | Choice | `heuristic` | Intelligence provider: `heuristic`, `gemini`, `openai`, `anthropic`, `ollama`, `local_weights`. |
| `--model-weights` | Path | `""` | Path to local model weights (GGUF file, Safetensors file, or HuggingFace directory). |
| `--device` | Choice | `auto` | Compute device for local weights: `auto`, `cuda`, `cpu`, or `mps`. |
| `--gpu-layers` | Integer | `-1` | Number of model layers to offload to GPU (`-1` for all layers). |
| `--context-window` | Integer | `4096` | Context window size in tokens for local weights. |

---

## 6. Cluster Schedulers & Deployment Guides

### Slurm HPC Submission
Save as [`slurm_job.sbatch`](batch_agent/templates/slurm_job.sbatch):
```bash
#!/bin/bash
#SBATCH --job-name=yoyo_ai
#SBATCH --output=yoyo_%j.out
#SBATCH --error=yoyo_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=batch

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export CI=true

python3 -m batch_agent.cli \
    --goal "Execute non-interactive pipeline, verify unit tests, and generate audit report" \
    --files "data/input.csv" "src/pipeline.py" \
    --workspace "${SLURM_SUBMIT_DIR:-$(pwd)}" \
    --sandbox-profile strict \
    --log-file "yoyo_audit_${SLURM_JOB_ID:-run}.txt" \
    --csv-log "yoyo_audit_${SLURM_JOB_ID:-run}.csv"

exit $?
```
Submit to Slurm:
```bash
sbatch batch_agent/templates/slurm_job.sbatch
```

### PBS / PBS Pro / OpenPBS / Torque Submission
Save as [`pbs_job.pbs`](batch_agent/templates/pbs_job.pbs):
```bash
#!/bin/bash
#PBS -N yoyo_ai
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -j oe
#PBS -o yoyo_%J.log
#PBS -V

set -euo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
fi

CLEAN_JOBID="${PBS_JOBID:-local_pbs_run}"
CLEAN_JOBID="${CLEAN_JOBID%%.*}"

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export CI=true

python3 -m batch_agent.cli \
    --goal "Autonomously execute batch pipeline, run verification checks, and record audit trail" \
    --workspace "$(pwd)" \
    --sandbox-profile strict \
    --log-file "yoyo_audit_pbs_${CLEAN_JOBID}.txt" \
    --csv-log "yoyo_audit_pbs_${CLEAN_JOBID}.csv"

exit $?
```
Submit to PBS:
```bash
qsub batch_agent/templates/pbs_job.pbs
```

### Portable Cron, Systemd & CI/CD Runner
Save as [`run_batch.sh`](batch_agent/templates/run_batch.sh):
```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-$(pwd)}"
GOAL="${2:-Autonomously analyze codebase, run verification tests, and produce audit report}"

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export CI=true

python3 -m batch_agent.cli \
    --goal "${GOAL}" \
    --workspace "${WORKSPACE}" \
    --sandbox-profile workspace \
    --log-file "${WORKSPACE}/yoyo_audit.txt" \
    --csv-log "${WORKSPACE}/yoyo_audit.csv"

exit $?
```

### JSON Batch Manifest Execution
Create a job manifest `job.json`:
```json
{
  "job_id": "batch-pipeline-001",
  "goal": "Process telemetry log records and generate aggregate summary report",
  "associated_files": [
    "data/telemetry.csv",
    "scripts/process_telemetry.py"
  ],
  "workspace_root": "./workspace",
  "max_steps": 15,
  "job_timeout_seconds": 600,
  "max_token_budget": 100000,
  "max_consecutive_failed_actions": 3,
  "enable_recovery_engine": true,
  "goal_supremacy": true,
  "provider": "heuristic",
  "sandbox": {
    "enabled": true,
    "profile": "workspace"
  },
  "process": {
    "default_timeout_seconds": 30,
    "enforce_non_interactive_stdin": true
  },
  "audit": {
    "log_file_path": "batch_audit.txt",
    "flush_immediately": true,
    "record_active_processes_snapshot": true
  }
}
```
Run the manifest:
```bash
python3 -m batch_agent.cli --manifest job.json
```

---

## 7. Audit Trail Sample Outputs

### Human-Readable .txt Audit Log Sample
Below is an excerpt from `batch_agent_audit.txt`:

```text
======================= YOYO - AUDIT TRAIL [JOB: batch-pipeline-001] =======================
Job ID             : batch-pipeline-001
Execution Mode     : NON-INTERACTIVE BATCH OS ENVIRONMENT
Start Timestamp    : 2026-11-10T19:04:09.361050+00:00
Workspace Root     : /path/to/workspace
Audit Log File     : /path/to/workspace/batch_audit_pipeline.txt
Sandbox Boundary   : ENFORCED (Profile: WORKSPACE)
Intelligence Model : heuristic
Goal Specification : Process telemetry log records, calculate aggregate statistics, and generate report
Associated Files   : [data/telemetry.csv, scripts/process_telemetry.py]
Max Allowed Steps  : 10
Safety Disclaimer  : YoYo can make mistakes. Always run on copies and maintain backups.
==========================================================================================

--- [2026-11-10T19:04:09.361579+00:00] [CHAIN_OF_THOUGHT] [STEP 1] -----------------------
Reasoning:
  Phase: INITIALIZATION
  Goal Analysis:
    Primary Goal: Process telemetry log records, calculate aggregate statistics, and generate report.
  Hypothesis:
    Associated input files exist and require structural inspection.
  Trade-off Evaluation:
    Direct inspection first ensures valid data formats before execution.
  Optimal Decision Taken:
    Inspect and read data/telemetry.csv to determine schema and row counts.

--- [2026-11-10T19:04:09.363842+00:00] [TOOL_INVOCATION] [STEP 1] ------------------------
Title       : Tool Invoked: file_read
Description : Read contents of data/telemetry.csv
  tool_name       : file_read
  target_file     : data/telemetry.csv
  sandbox_check   : PASS (Target inside workspace boundary)

--- [2026-11-10T19:04:09.510342+00:00] [RECOVERY_ACTION] [STEP 3] ------------------------
Title       : Autonomous Command Recovery: WORKSPACE_PATH_RELINKING
Description : Command 'python3 calc.py' failed with file not found. Auto-relinked to 'python3 src/calc.py'.
  status          : RECOVERED
  strategy        : relink_workspace_path
  repaired_cmd    : python3 src/calc.py

--- [2026-11-10T19:04:09.608217+00:00] [JOB_COMPLETION] ----------------------------------
Title       : Batch Job Finished: SUCCESS
Description : YoYo successfully completed goal across 6 steps.
  status          : SUCCESS
  total_steps     : 6
  total_runtime_seconds: 0.25
  artifacts       :
    - batch_summary.md

Active Subprocesses Snapshot:
  [No active background child processes]

========================== AUDIT TRAIL COMPLETE - FINAL SUMMARY ==========================
Final Job Status   : SUCCESS
Total Steps Taken  : 6
Total Job Runtime  : 0.25 seconds
Produced Artifacts : batch_summary.md
Summary            : YoYo successfully completed goal across 6 steps.
Active Subprocesses: 0 remaining (Cleaned up by YoYo)
==========================================================================================
```

### Machine-Parseable .csv Audit Log Sample
Below is an excerpt from `batch_agent_audit.csv`:

```csv
timestamp,step,event_type,title,status,exit_code,duration_seconds,command,tokens_used,active_pids,error_summary,description
2026-11-10T19:04:09.361050Z,0,JOB_START,Batch Job Ingestion & Environment Initialization,SUCCESS,0,0.001,,0,0,,Ingested goal and associated files into non-interactive batch sandbox.
2026-11-10T19:04:09.361579Z,1,CHAIN_OF_THOUGHT,Chain of Thought Reasoning,SUCCESS,0,0.001,,150,0,,Phase: INITIALIZATION. Inspect and read data/telemetry.csv
2026-11-10T19:04:09.363842Z,1,TOOL_INVOCATION,Tool Invoked: file_read,SUCCESS,0,0.002,,180,0,,Read contents of data/telemetry.csv
2026-11-10T19:04:09.450123Z,2,COMMAND_EXECUTION,Command Execution: python3 process_telemetry.py,SUCCESS,0,0.124,python3 process_telemetry.py,320,0,,Process completed with exit code 0
2026-11-10T19:04:09.510342Z,3,RECOVERY_ACTION,Autonomous Command Recovery: WORKSPACE_PATH_RELINKING,SUCCESS,0,0.045,python3 src/calc.py,450,0,,Command relinked to src/calc.py
2026-11-10T19:04:09.608217Z,6,JOB_COMPLETION,Batch Job Finished: SUCCESS,SUCCESS,0,0.247,,620,0,,YoYo successfully completed goal across 6 steps.
```

---

## 8. Verification & Comprehensive Test Suite

YoYo is validated by a rigorous test suite of **64 unit and end-to-end integration tests** covering every functional subsystem:

1. **`test_sandbox.py`**: Workspace boundary confinement, symlink traversal prevention, sensitive directory/file read/write filtering, dangerous command blocking (`sudo`, fork bomb, `rm -rf /`), and environment variable secret scrubbing.
2. **`test_process_tracker.py`**: Non-interactive `DEVNULL` stdin enforcement, process group isolation, active process tracking table, execution timeout killing, and emergency cleanups.
3. **`test_audit_logger.py`**: Synchronous disk flushing, structured Chain of Thought formatting, command execution output capture, active process logging, and preemption survival.
4. **`test_tools.py`**: Non-interactive filesystem operations, multi-stage resilient fuzzy patcher (exact, whitespace-normalized, indentation-shifted, and sequence matching), and automated syntax verification.
5. **`test_autonomous_engine.py`**: Complete autonomous end-to-end goal execution without human intervention, plan step progression, automated verification, and final artifact generation.
6. **`test_cli.py`**: Command-line flag parsing, JSON manifest ingestion, workspace auto-detection, and batch return codes.
7. **`test_checkpoint.py`**: Crash-resilient checkpointing (`.batch_checkpoint.json`), state persistence with synchronous `fsync`, and seamless resumption from prior checkpoints.
8. **`test_offline_planner.py`**: 2-stage air-gapped workflow, autonomous environment survey, `plan.txt` generation with dependency rationale, `proposal/` code drafting, and Stage 2 execution.
9. **`test_local_weights.py`**: Direct model weights connection, GGUF parameter validation, and AST symbol inspection tools.
10. **`test_goal_priority_and_replanning.py`**: Goal supremacy mandate, dynamic plan adaptation, inserted verification steps, and plan mutation recording.
11. **`test_runaway_loop_and_recovery.py`**: Hard token budgets, infinite retry action circuit breakers, command failure diagnosis and self-healing recovery, dual CSV logging, and fully toggleable runaway loop defense mechanisms.
12. **`test_document_ingestion.py`**: Pre-flight research paper ingestion (PDF, LaTeX, Markdown) and SRS requirements extraction.
13. **`test_production_readiness.py`**: `.gitignore`/`.venv`/`node_modules`/`build` directory pruning, `allow_dummy_stubs=False` enforcement, cross-step working memory, C/C++ structural AST inspection, GoogleTest/CTest/Catch2/Go test failure classification, and on-disk drill-down for huge single files (`file_search` `path` + `context_lines`, `ast_inspect` `query`, ranged `file_read`).

### Running the Test Suite
```bash
python3 -m unittest discover -s tests -v
```

**Result**: **64 / 64 tests pass cleanly** in ~5 seconds with zero external dependencies.

### Live Scale Benchmark (300,038 LOC / 10,715 Files)
```bash
GEMINI_API_KEY="your-key" python3 examples/real_code_fix_benchmark.py --loc 300000 --provider gemini
```
Verified end-to-end with live Gemini API calls: located a buried defect across **10,715 files (300,038 LOC)**, patched it on disk, verified passing unit tests, and exited `SUCCESS (0)` while consuming **11,844 total tokens** ($O(1)$ scaling).
