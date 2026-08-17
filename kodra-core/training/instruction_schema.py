"""
Phase 3 instruction-tuning data schema for Kodra AI Agent.

This module defines the FORMAT that future instruction-tuning examples will
use. It does not perform any instruction tuning itself and Phase 1/2
checkpoints are never loaded through this path. Instruction tuning happens
only after base pretraining (Phase 1/2) is validated on real hardware.

Chat format follows a role-tagged transcript, matching the structure used by
most modern instruction-tuned coding assistants:

    <system>...</system><user>...</user><assistant>...</assistant>

Multiple turns are supported by repeating <user>/<assistant> pairs.
"""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = "You are Kodra AI Agent, an AI coding assistant."


class TaskCategory(str, Enum):
    CODE_GENERATION = "code_generation"
    EXPLANATION = "explanation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    REPOSITORY_QUESTIONS = "repository_questions"


@dataclass
class ChatTurn:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class InstructionExample:
    id: str
    category: TaskCategory
    language: Optional[str]
    turns: List[ChatTurn]
    source: str = "synthetic"  # provenance: synthetic | curated | licensed-dataset:<name>
    license: str = "unknown"

    def to_prompt_string(self) -> str:
        """Serializes the example to the flat tagged-transcript format the
        tokenizer will eventually see during instruction tuning."""
        parts = []
        for turn in self.turns:
            parts.append(f"<{turn.role}>\n{turn.content}\n</{turn.role}>")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


def make_example(
    id: str,
    category: TaskCategory,
    user_message: str,
    assistant_message: str,
    language: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
    source: str = "synthetic",
    license: str = "unknown",
) -> InstructionExample:
    return InstructionExample(
        id=id,
        category=category,
        language=language,
        source=source,
        license=license,
        turns=[
            ChatTurn("system", system_prompt),
            ChatTurn("user", user_message),
            ChatTurn("assistant", assistant_message),
        ],
    )


# A handful of schema-conformant EXAMPLES (not a training set) demonstrating
# every category, so the format is testable and self-documenting. Phase 3
# will replace/extend these with a real, much larger curated dataset.
SCHEMA_EXAMPLES: List[InstructionExample] = [
    make_example(
        "example_debug_1", TaskCategory.DEBUGGING,
        "Fix the bug in this Python function:\n\ndef add(a, b):\n    return a - b",
        "The function subtracts instead of adding. Fixed version:\n\ndef add(a, b):\n    return a + b",
        language="python",
    ),
    make_example(
        "example_gen_1", TaskCategory.CODE_GENERATION,
        "Write a function that reverses a string in Python.",
        "def reverse_string(s: str) -> str:\n    return s[::-1]",
        language="python",
    ),
    make_example(
        "example_test_1", TaskCategory.TEST_GENERATION,
        "Write a unit test for this function:\n\ndef add(a, b):\n    return a + b",
        "import unittest\n\nclass TestAdd(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)",
        language="python",
    ),
    make_example(
        "example_explain_1", TaskCategory.EXPLANATION,
        "Explain what this code does:\n\n[x for x in range(10) if x % 2 == 0]",
        "This is a list comprehension that produces all even numbers from 0 to 9.",
        language="python",
    ),
    make_example(
        "example_refactor_1", TaskCategory.REFACTORING,
        "Refactor this to avoid the nested if:\n\nif a:\n    if b:\n        do_thing()",
        "if a and b:\n    do_thing()",
        language="python",
    ),
    make_example(
        "example_docs_1", TaskCategory.DOCUMENTATION,
        "Write a docstring for:\n\ndef add(a, b):\n    return a + b",
        '"""Return the sum of a and b."""',
        language="python",
    ),
    make_example(
        "example_repo_1", TaskCategory.REPOSITORY_QUESTIONS,
        "Where is the KodraGPT model class defined?",
        "In kodra-core/model/gpt_model.py, as the `KodraGPT` class.",
        language=None,
    ),
]
