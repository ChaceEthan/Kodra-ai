"""
Kodra Agent tool foundation (roadmap scaffolding, Phase 3+).

These classes define the SAFE INTERFACE that a future Kodra Agent tool-use
loop will call into. None of this is wired to a live LLM tool-call loop yet
- Kodra GPT Phase 1 is a plain causal language model with no tool-calling
head or agent runtime. This module exists so the interface is designed and
tested ahead of time.

Safety model:
  - Read-only tools (read_file, search_files, search_code, list_directory,
    git_status, git_diff) execute immediately - they cannot mutate anything.
  - Write/mutating tools (create_file, edit_file, apply_patch) and the
    terminal/test tools (run_terminal, run_tests) are NOT auto-executed.
    Calling them returns a ToolResult with status="requires_approval" and
    never touches disk or spawns a process until `approved=True` is passed
    explicitly by the caller (i.e. a human-in-the-loop confirmation).
  - `run_terminal` additionally refuses a fixed denylist of destructive
    command patterns (rm -rf, git push --force, format, del /s, etc.) even
    when approved=True, because a coding agent should never be able to talk
    itself into running those.
"""
import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    REQUIRES_APPROVAL = "requires_approval"
    DENIED = "denied"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass
class ToolResult:
    status: ToolStatus
    data: Any = None
    message: str = ""


# Command substrings that run_terminal refuses even with approved=True.
_DENYLIST_PATTERNS = [
    r"\brm\s+-rf\b", r"\bdel\s+/s\b", r"\bformat\s+[a-zA-Z]:", r"\bmkfs\b",
    r"git\s+push\s+.*--force", r"git\s+reset\s+--hard", r":\(\)\{.*\};:",  # fork bomb
    r"\bshutdown\b", r"\breboot\b", r"\bdd\s+if=",
]


class KodraAgentTools:
    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)

    def _resolve_safe_path(self, relative_path: str) -> Optional[str]:
        full = os.path.abspath(os.path.join(self.workspace_root, relative_path))
        if not full.startswith(self.workspace_root):
            return None  # path traversal outside workspace - refuse
        return full

    # --- Read-only tools (execute immediately) ---------------------------
    def read_file(self, relative_path: str, max_bytes: int = 200_000) -> ToolResult:
        full = self._resolve_safe_path(relative_path)
        if full is None:
            return ToolResult(ToolStatus.DENIED, message="Path escapes workspace root")
        if not os.path.exists(full):
            return ToolResult(ToolStatus.ERROR, message="File not found")
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_bytes)
            return ToolResult(ToolStatus.OK, data=content)
        except OSError as e:
            return ToolResult(ToolStatus.ERROR, message=str(e))

    def list_directory(self, relative_path: str = ".") -> ToolResult:
        full = self._resolve_safe_path(relative_path)
        if full is None:
            return ToolResult(ToolStatus.DENIED, message="Path escapes workspace root")
        if not os.path.isdir(full):
            return ToolResult(ToolStatus.ERROR, message="Not a directory")
        return ToolResult(ToolStatus.OK, data=sorted(os.listdir(full)))

    def search_files(self, glob_pattern: str) -> ToolResult:
        matches = []
        for dirpath, _dirnames, filenames in os.walk(self.workspace_root):
            for fname in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fname), self.workspace_root)
                if fnmatch.fnmatch(rel.replace(os.sep, "/"), glob_pattern):
                    matches.append(rel.replace(os.sep, "/"))
        return ToolResult(ToolStatus.OK, data=sorted(matches))

    def search_code(self, pattern: str, relative_path: str = ".", max_matches: int = 200) -> ToolResult:
        full = self._resolve_safe_path(relative_path)
        if full is None:
            return ToolResult(ToolStatus.DENIED, message="Path escapes workspace root")
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(ToolStatus.ERROR, message=f"Invalid regex: {e}")

        results: List[Dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__", ".venv")]
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, start=1):
                            if regex.search(line):
                                results.append({
                                    "file": os.path.relpath(fpath, self.workspace_root).replace(os.sep, "/"),
                                    "line": lineno, "text": line.strip(),
                                })
                                if len(results) >= max_matches:
                                    return ToolResult(ToolStatus.OK, data=results)
                except OSError:
                    continue
        return ToolResult(ToolStatus.OK, data=results)

    def git_status(self) -> ToolResult:
        return self._run_readonly_git(["status", "--porcelain"])

    def git_diff(self) -> ToolResult:
        return self._run_readonly_git(["diff"])

    def _run_readonly_git(self, args: List[str]) -> ToolResult:
        try:
            res = subprocess.run(["git"] + args, cwd=self.workspace_root, capture_output=True, text=True, timeout=30)
            return ToolResult(ToolStatus.OK, data=res.stdout, message=res.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            return ToolResult(ToolStatus.ERROR, message=str(e))

    # --- Mutating tools (require explicit human approval) -----------------
    def create_file(self, relative_path: str, content: str, approved: bool = False) -> ToolResult:
        if not approved:
            return ToolResult(ToolStatus.REQUIRES_APPROVAL, message="create_file requires explicit approval")
        full = self._resolve_safe_path(relative_path)
        if full is None:
            return ToolResult(ToolStatus.DENIED, message="Path escapes workspace root")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(ToolStatus.OK, message=f"Created {relative_path}")

    def edit_file(self, relative_path: str, new_content: str, approved: bool = False) -> ToolResult:
        if not approved:
            return ToolResult(ToolStatus.REQUIRES_APPROVAL, message="edit_file requires explicit approval")
        full = self._resolve_safe_path(relative_path)
        if full is None or not os.path.exists(full):
            return ToolResult(ToolStatus.ERROR, message="File not found or path escapes workspace")
        with open(full, "w", encoding="utf-8") as f:
            f.write(new_content)
        return ToolResult(ToolStatus.OK, message=f"Edited {relative_path}")

    def apply_patch(self, _unified_diff: str, approved: bool = False) -> ToolResult:
        # Roadmap placeholder: real unified-diff application is not implemented yet.
        return ToolResult(ToolStatus.NOT_IMPLEMENTED, message="apply_patch is a roadmap placeholder")

    def run_tests(self, test_path: str = "tests/", approved: bool = False) -> ToolResult:
        if not approved:
            return ToolResult(ToolStatus.REQUIRES_APPROVAL, message="run_tests requires explicit approval")
        try:
            res = subprocess.run(
                ["python", "-m", "pytest", test_path, "-q"],
                cwd=self.workspace_root, capture_output=True, text=True, timeout=600,
            )
            return ToolResult(ToolStatus.OK, data=res.stdout, message=res.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            return ToolResult(ToolStatus.ERROR, message=str(e))

    def run_terminal(self, command: str, approved: bool = False) -> ToolResult:
        if not approved:
            return ToolResult(ToolStatus.REQUIRES_APPROVAL, message="run_terminal requires explicit approval")
        for pattern in _DENYLIST_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return ToolResult(ToolStatus.DENIED, message=f"Command matches denylisted destructive pattern: {pattern}")
        try:
            res = subprocess.run(command, shell=True, cwd=self.workspace_root, capture_output=True, text=True, timeout=120)
            return ToolResult(ToolStatus.OK, data=res.stdout, message=res.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            return ToolResult(ToolStatus.ERROR, message=str(e))
