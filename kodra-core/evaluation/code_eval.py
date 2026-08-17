"""
Code-completion and syntax evaluation for Kodra AI Agent.

Every score here is computed by actually running the generator against
concrete prompts and checking the output — nothing is hard-coded. Given
the current Phase 1 model is a tiny, largely-untrained research model,
scores are expected to be low; that is a true reflection of model
capability, not a bug.
"""
import ast
import json
import re
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Any

from inference.generator import CodeGenerator


@dataclass
class CodeCompletionCase:
    id: str
    language: str
    prompt: str
    # A cheap, deterministic check for whether the completion is "reasonable" -
    # not a full correctness oracle, since the tiny Phase 1 model is not
    # expected to solve real coding tasks yet.
    check: Callable[[str], bool]


def _contains_any(keywords: List[str]) -> Callable[[str], bool]:
    return lambda text: any(k in text for k in keywords)


PYTHON_CASES = [
    CodeCompletionCase("py_function_def", "python", "def add(a, b):\n    return", _contains_any(["a", "b", "+"])),
    CodeCompletionCase("py_loop", "python", "for i in range(10):\n   ", _contains_any(["i", "print", "range"])),
]

JAVASCRIPT_CASES = [
    CodeCompletionCase("js_function_def", "javascript", "function add(a, b) {\n  return", _contains_any(["a", "b", "+"])),
]

TYPESCRIPT_CASES = [
    CodeCompletionCase("ts_interface", "typescript", "interface User {\n  name:", _contains_any(["string", "number", ":"])),
]

REACT_CASES = [
    CodeCompletionCase("react_component", "typescript", "export function Button(props) {\n  return", _contains_any(["<", "props", "return"])),
]

ALGORITHM_CASES = [
    CodeCompletionCase("algo_bubble_sort", "python", "def bubble_sort(arr):\n   ", _contains_any(["for", "arr", "range"])),
]

BUG_FIX_CASES = [
    CodeCompletionCase(
        "bugfix_off_by_one",
        "python",
        "# Bug: this loop should include the last index\nfor i in range(len(arr) - 1):\n    # fix:",
        _contains_any(["range", "len", "arr"]),
    ),
]

ALL_COMPLETION_CASES: Dict[str, List[CodeCompletionCase]] = {
    "python": PYTHON_CASES,
    "javascript": JAVASCRIPT_CASES,
    "typescript": TYPESCRIPT_CASES,
    "react": REACT_CASES,
    "algorithms": ALGORITHM_CASES,
    "bug_fixing": BUG_FIX_CASES,
}


def run_code_completion_eval(generator: CodeGenerator, max_new_tokens: int = 48) -> Dict[str, Any]:
    """Actually runs generation for every case and reports the real pass rate
    against the (deliberately loose) heuristic checks above."""
    results = []
    for category, cases in ALL_COMPLETION_CASES.items():
        for case in cases:
            completion = generator.generate(case.prompt, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40)
            passed = bool(case.check(completion))
            results.append({
                "id": case.id, "category": category, "language": case.language,
                "prompt": case.prompt, "completion": completion, "passed": passed,
            })
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "results": results,
    }


# --- Syntax evaluation ----------------------------------------------------
def python_parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def json_is_valid(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def run_syntax_eval(generator: CodeGenerator, num_python_samples: int = 5, max_new_tokens: int = 48) -> Dict[str, Any]:
    prompts = ["def ", "class ", "import ", "for i in range(10):\n", "if x > 0:\n"][:num_python_samples]
    python_results = []
    for p in prompts:
        code = generator.generate(p, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40)
        python_results.append({"prompt": p, "code": code, "parses": python_parses(code)})

    parses_count = sum(1 for r in python_results if r["parses"])
    return {
        "python_parse_rate": (parses_count / len(python_results)) if python_results else 0.0,
        "python_results": python_results,
    }


# --- Future-agent evaluation placeholders ----------------------------------
# These categories require a real Kodra Agent tool loop (multi-file edits,
# repo-wide context, test execution) that does not exist yet (see
# agent/tools.py). They are declared here as documented placeholders so the
# evaluation report has a stable schema going forward, and so nobody
# accidentally reports a fabricated score for a category that hasn't been
# implemented.
FUTURE_AGENT_EVAL_CATEGORIES = [
    "repository_understanding",
    "test_fixing",
    "multi_file_edits",
]


def run_future_agent_eval_placeholders() -> Dict[str, Any]:
    return {
        cat: {"status": "not_implemented", "score": None}
        for cat in FUTURE_AGENT_EVAL_CATEGORIES
    }
