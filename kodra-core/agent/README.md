# Kodra Agent

Roadmap directory for Kodra Agent (Phase 2 & Phase 3).

**KODRA AGENT RUNTIME: NOT YET COMPLETE.** Individual tools work and are
tested, but there is no planner turning Kodra GPT output into tool calls,
so there is no working autonomous agent loop. See `GET /api/agent/status`
for live status.

## What's implemented

| Module | Status |
|---|---|
| `tools.py` | All 11 tools implemented and tested: `read_file`, `search_files`, `search_code`, `list_directory`, `git_status`, `git_diff` (read-only, execute immediately); `create_file`, `edit_file`, `apply_patch`, `run_tests`, `run_terminal` (mutating, approval-gated) |
| `tool_registry.py` | Static catalog of the tools above and whether each mutates |
| `permissions.py` | `PermissionPolicy` - the single place that decides ALLOW / REQUIRES_APPROVAL / DENY for a tool call |
| `context.py` | `AgentContext` - shared state container (workspace, tools, permissions) |
| `runtime.py` | `AgentRuntime.call_tool()` - real, safe direct tool dispatch through the permission policy. `run_autonomous_task()` explicitly returns `NOT_YET_COMPLETE` |
| `planner.py` | `Planner.plan_next_step()` - roadmap placeholder, raises `NotImplementedError` |

## Safety model

- Mutating tools require explicit human approval by default
  (`KODRA_REQUIRE_TOOL_APPROVAL=true`).
- `run_terminal` is disabled by default (`KODRA_ENABLE_TERMINAL_TOOLS=false`)
  and, even when enabled, always requires approval and is filtered against a
  denylist of destructive command patterns.
- **Autonomous mutation is refused unless a trained checkpoint is loaded.**
  A randomly-initialized model calling a mutating tool with nobody in the
  loop is a hard `DENY` in `PermissionPolicy`, independent of any approval
  flag - see `test_agent_runtime.py::TestPermissionPolicy`.

## Not implemented

- The autonomous loop itself (model → planner → tool selection → permission
  check → execution → observation → next step → final answer).
- Repository context / retrieval (`REPOSITORY VECTOR INDEX: NOT_YET_IMPLEMENTED`).
  `VECTOR_DB_PROVIDER` / `VECTOR_DB_PATH` are a configuration boundary only;
  nothing reads or writes embeddings today.
