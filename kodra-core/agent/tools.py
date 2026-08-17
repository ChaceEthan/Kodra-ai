"""
Kodra Agent tool foundation (roadmap scaffolding, Phase 3+).

These classes define the SAFE INTERFACE that a future Kodra Agent tool-use
loop will call into. None of this is wired to a live LLM tool-call loop yet
- Kodra GPT Phase 1 is a plain causal language model with no tool-calling
head or agent runtime. This module exists so the interface is designed and
tested ahead of time.

Future planner-loop interface boundary (not implemented - see
agent/runtime.py, agent/planner.py, agent/permissions.py, agent/context.py,
and agent/tool_registry.py): model -> planner -> tool request -> approval
-> tool execution -> observation -> model continuation.

Safety model:
  - Read-only tools (read_file, search_files, search_code, list_directory,
    git_status, git_diff) execute immediately - they cannot mutate anything.
  - Write/mutating tools (create_file, edit_file, apply_patch, run_tests)
    are gated by `require_approval` (from KODRA_REQUIRE_TOOL_APPROVAL,
    default True). With approval required and not granted, they return a
    ToolResult with status="requires_approval" and never touch disk.
  - `run_terminal` is additionally gated by `enable_terminal_tools` (from
    KODRA_ENABLE_TERMINAL_TOOLS, default False) and ALWAYS requires
    approval regardless of `require_approval` - this cannot be configured
    away, because a coding agent should never be able to talk itself into
    unrestricted shell execution.
  - `run_terminal` also refuses a fixed denylist of destructive command
    patterns (rm -rf, git push --force, format, del /s, etc.) even when
    approved=True.
"""
import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from configs.env import load_runtime_config


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


def resolve_within_root(root: str, relative_path: str) -> Optional[str]:
    """Resolves `relative_path` against `root` and verifies the canonical
    result is actually inside `root` - not merely string-prefixed by it.
    A plain `full.startswith(root)` check is bypassable by a sibling
    directory that happens to share root's name as a prefix (e.g. root
    "/work/kodra-core" vs sibling "/work/kodra-core-secrets"), so this also
    requires a path separator (or exact equality) right after the root."""
    resolved_root = os.path.abspath(root)
    full = os.path.abspath(os.path.join(resolved_root, relative_path))
    if full != resolved_root and not full.startswith(resolved_root + os.sep):
        return None
    return full


class KodraAgentTools:
    def __init__(
        self,
        workspace_root: str,
        require_approval: Optional[bool] = None,
        enable_terminal_tools: Optional[bool] = None,
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        runtime = load_runtime_config()
        self.require_approval = runtime.require_tool_approval if require_approval is None else require_approval
        self.enable_terminal_tools = runtime.enable_terminal_tools if enable_terminal_tools is None else enable_terminal_tools

    def _resolve_safe_path(self, relative_path: str) -> Optional[str]:
        return resolve_within_root(self.workspace_root, relative_path)

    def _needs_approval(self, approved: bool) -> bool:
        return self.require_approval and not approved

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
        if self._needs_approval(approved):
            return ToolResult(ToolStatus.REQUIRES_APPROVAL, message="create_file requires explicit approval")
        full = self._resolve_safe_path(relative_path)
        if full is None:
            return ToolResult(ToolStatus.DENIED, message="Path escapes workspace root")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(ToolStatus.OK, message=f"Created {relative_path}")

    def edit_file(self, relative_path: str, new_content: str, approved: bool = False) -> ToolResult:
        if self._needs_approval(approved):
            return ToolResult(ToolStatus.REQUIRES_APPROVAL, message="edit_file requires explicit approval")
        full = self._resolve_safe_path(relative_path)
        if full is None or not os.path.exists(full):
            return ToolResult(ToolStatus.ERROR, message="File not found or path escapes workspace")
        with open(full, "w", encoding="utf-8") as f:
            f.write(new_content)
        return ToolResult(ToolStatus.OK, message=f"Edited {relative_path}")

    def apply_patch(self, unified_diff: str, approved: bool = False, dry_run: bool = False) -> ToolResult:
        """Applies a unified diff to one or more existing files within the
        workspace. Supports a dry-run preview (no writes), requires explicit
        approval to actually write, and returns the pre-patch content of
        every touched file so the caller can revert by writing it back."""
        try:
            per_file_hunks = _parse_unified_diff(unified_diff)
        except ValueError as e:
            return ToolResult(ToolStatus.ERROR, message=f"Could not parse patch: {e}")

        if not per_file_hunks:
            return ToolResult(ToolStatus.ERROR, message="Patch contained no recognizable file hunks")

        previews: Dict[str, Dict[str, str]] = {}
        for rel_path, hunks in per_file_hunks.items():
            full = self._resolve_safe_path(rel_path)
            if full is None:
                return ToolResult(ToolStatus.DENIED, message=f"Patch target escapes workspace root: {rel_path}")
            if not os.path.exists(full):
                return ToolResult(ToolStatus.ERROR, message=f"apply_patch only supports existing files, not found: {rel_path}")
            with open(full, "r", encoding="utf-8") as f:
                original = f.read()
            try:
                new_content = _apply_hunks(original, hunks)
            except ValueError as e:
                return ToolResult(ToolStatus.ERROR, message=f"Patch does not apply cleanly to {rel_path}: {e}")
            previews[rel_path] = {"original": original, "patched": new_content}

        if dry_run or self._needs_approval(approved):
            status = ToolStatus.OK if dry_run else ToolStatus.REQUIRES_APPROVAL
            return ToolResult(
                status,
                data={"preview": {p: v["patched"] for p, v in previews.items()}},
                message="Dry run - no files were written" if dry_run else "apply_patch requires explicit approval",
            )

        backups: Dict[str, str] = {}
        for rel_path, contents in previews.items():
            full = self._resolve_safe_path(rel_path)
            backups[rel_path] = contents["original"]
            with open(full, "w", encoding="utf-8") as f:
                f.write(contents["patched"])

        return ToolResult(
            ToolStatus.OK,
            data={"applied_files": list(previews.keys()), "backups": backups},
            message=f"Applied patch to {len(previews)} file(s). Original content returned in data.backups for reversal.",
        )

    def run_tests(self, test_path: str = "tests/", approved: bool = False) -> ToolResult:
        if self._needs_approval(approved):
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
        if not self.enable_terminal_tools:
            return ToolResult(ToolStatus.DENIED, message="Terminal tools are disabled (KODRA_ENABLE_TERMINAL_TOOLS=false)")
        if not approved:
            # Always requires approval, independent of require_approval -
            # shell execution is never allowed to skip human confirmation.
            return ToolResult(ToolStatus.REQUIRES_APPROVAL, message="run_terminal requires explicit approval")
        for pattern in _DENYLIST_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return ToolResult(ToolStatus.DENIED, message=f"Command matches denylisted destructive pattern: {pattern}")
        try:
            res = subprocess.run(command, shell=True, cwd=self.workspace_root, capture_output=True, text=True, timeout=120)
            return ToolResult(ToolStatus.OK, data=res.stdout, message=res.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            return ToolResult(ToolStatus.ERROR, message=str(e))


# --- Minimal unified-diff parser/applier (stdlib only) ----------------------
@dataclass
class _Hunk:
    old_start: int
    old_lines: List[str]
    new_lines: List[str]


_FILE_HEADER_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def _parse_unified_diff(diff_text: str) -> Dict[str, List["_Hunk"]]:
    files: Dict[str, List[_Hunk]] = {}
    current_file: Optional[str] = None
    current_hunk: Optional[_Hunk] = None

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            continue
        m = _FILE_HEADER_RE.match(line)
        if m:
            current_file = m.group(1).strip()
            files.setdefault(current_file, [])
            current_hunk = None
            continue
        m = _HUNK_HEADER_RE.match(line)
        if m:
            if current_file is None:
                raise ValueError("Hunk header found before a +++ file header")
            current_hunk = _Hunk(old_start=int(m.group(1)), old_lines=[], new_lines=[])
            files[current_file].append(current_hunk)
            continue
        if current_hunk is None:
            continue
        if line.startswith("+"):
            current_hunk.new_lines.append(line[1:])
        elif line.startswith("-"):
            current_hunk.old_lines.append(line[1:])
        elif line.startswith(" "):
            current_hunk.old_lines.append(line[1:])
            current_hunk.new_lines.append(line[1:])
        # lines like "\ No newline at end of file" are ignored

    return files


def _apply_hunks(original: str, hunks: List["_Hunk"]) -> str:
    lines = original.split("\n")
    offset = 0
    for hunk in hunks:
        start_idx = hunk.old_start - 1 + offset
        end_idx = start_idx + len(hunk.old_lines)
        if start_idx < 0 or end_idx > len(lines):
            raise ValueError("hunk out of range")
        actual = lines[start_idx:end_idx]
        if actual != hunk.old_lines:
            raise ValueError(f"context mismatch at line {hunk.old_start}")
        lines[start_idx:end_idx] = hunk.new_lines
        offset += len(hunk.new_lines) - len(hunk.old_lines)
    return "\n".join(lines)

