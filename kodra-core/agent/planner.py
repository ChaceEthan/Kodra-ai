"""
Planner interface boundary (roadmap placeholder - not implemented).

A real planner takes the model's raw output and turns it into a structured
ToolRequest (or a final answer). That requires an instruction-tuned model
that actually emits tool calls in a parseable format. Kodra GPT Phase 1 is
a plain causal language model with no tool-calling head and no instruction
tuning (see training/instruction_schema.py for the Phase 3 data format that
would eventually make this possible) - so there is nothing for a planner to
parse yet.

This class exists purely to give the future runtime a stable interface to
call, and to make the "not implemented" status explicit and testable rather
than silently absent.
"""
from agent.runtime import ToolRequest


class Planner:
    def __init__(self, model_can_emit_tool_calls: bool = False):
        self.model_can_emit_tool_calls = model_can_emit_tool_calls

    def plan_next_step(self, model_output: str) -> ToolRequest:
        if not self.model_can_emit_tool_calls:
            raise NotImplementedError(
                "Planner.plan_next_step is a roadmap placeholder: Kodra GPT "
                "does not yet emit structured tool calls (no instruction "
                "tuning / tool-calling head). See training/instruction_schema.py "
                "for the Phase 3 data format this depends on."
            )
        raise NotImplementedError("Tool-call parsing is not implemented yet.")
