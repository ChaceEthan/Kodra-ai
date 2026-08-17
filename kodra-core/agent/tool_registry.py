"""
Static catalog of the tools KodraAgentTools implements, for use by the
future planner/runtime. This is metadata only - it does not execute
anything. See agent/tools.py for the actual implementations.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    mutating: bool  # True if this tool can write to disk or spawn a shell
    requires_terminal: bool = False  # True only for run_terminal


TOOL_REGISTRY: Dict[str, ToolDefinition] = {
    "read_file": ToolDefinition("read_file", "Read a file's contents within the workspace.", mutating=False),
    "list_directory": ToolDefinition("list_directory", "List entries in a workspace directory.", mutating=False),
    "search_files": ToolDefinition("search_files", "Find files by glob pattern within the workspace.", mutating=False),
    "search_code": ToolDefinition("search_code", "Regex-search file contents within the workspace.", mutating=False),
    "git_status": ToolDefinition("git_status", "Show working-tree status (read-only).", mutating=False),
    "git_diff": ToolDefinition("git_diff", "Show the working-tree diff (read-only).", mutating=False),
    "create_file": ToolDefinition("create_file", "Create a new file with given content.", mutating=True),
    "edit_file": ToolDefinition("edit_file", "Overwrite an existing file's content.", mutating=True),
    "apply_patch": ToolDefinition("apply_patch", "Apply a unified diff to existing file(s), with dry-run and reversible backups.", mutating=True),
    "run_tests": ToolDefinition("run_tests", "Run the pytest suite.", mutating=True),
    "run_terminal": ToolDefinition("run_terminal", "Run an arbitrary shell command (denylist-filtered).", mutating=True, requires_terminal=True),
}


def list_tools() -> List[ToolDefinition]:
    return list(TOOL_REGISTRY.values())


def get_tool(name: str) -> ToolDefinition:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool '{name}'. Available: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name]
