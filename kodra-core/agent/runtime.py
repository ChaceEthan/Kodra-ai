"""
Kodra Agent runtime (roadmap scaffolding - the full autonomous loop is NOT
implemented).

The intended loop is:

    KodraGPT -> planner -> tool selection -> permission check
             -> tool execution -> observation -> next model step
             -> final answer

KODRA_AGENT_RUNTIME_STATUS below is the single source of truth for whether
that loop exists. It does not: Kodra GPT Phase 1 has no tool-calling head,
so there is no way for the model to select a tool in the first place (see
agent/planner.py). Do not present Kodra AI Agent as a working autonomous
coding agent while this says NOT_YET_COMPLETE.

What IS real and usable today:
  - Every individual tool in agent/tools.py works and is tested.
  - `AgentRuntime.call_tool` lets a human (e.g. via a future UI) invoke any
    tool directly, still gated by the same PermissionPolicy that a real
    autonomous loop would use. Read-only tools execute immediately;
    mutating tools still require explicit approval; autonomous mutation
    (i.e. the model deciding to call a mutating tool with nobody in the
    loop) is refused unless a trained checkpoint is loaded.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from agent.context import AgentContext
from agent.permissions import PermissionDecision
from agent.tool_registry import get_tool
from agent.tools import ToolResult, ToolStatus

KODRA_AGENT_RUNTIME_STATUS = "NOT_YET_COMPLETE"


class AgentStepKind(str, Enum):
    TOOL_REQUEST = "tool_request"
    OBSERVATION = "observation"
    MODEL_CONTINUATION = "model_continuation"


@dataclass
class ToolRequest:
    """A tool call the planner wants to make, before approval/execution."""
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class AgentStep:
    """One step in the future model<->tool loop. `kind` distinguishes a
    pending tool request from its resulting observation or the model's next
    continuation, so a transcript of steps fully reconstructs the loop."""
    kind: AgentStepKind
    tool_request: Optional[ToolRequest] = None
    tool_result: Optional[ToolResult] = None
    model_text: Optional[str] = None


@dataclass
class AgentRunResult:
    status: str
    message: str
    tool_result: Optional[ToolResult] = None


class AgentRuntime:
    def __init__(self, context: AgentContext):
        self.context = context

    def call_tool(self, tool_name: str, approved: bool = False, autonomous: bool = False, **kwargs: Any) -> AgentRunResult:
        """Dispatches a single tool call through the permission policy. This
        is the safe, real path for invoking tools today - it is NOT the
        full autonomous loop (see run_autonomous_task below)."""
        try:
            decision = self.context.permissions.decide(tool_name, approved=approved, autonomous=autonomous)
        except KeyError as e:
            return AgentRunResult(status="error", message=str(e))

        if decision == PermissionDecision.DENY:
            return AgentRunResult(status="denied", message=f"Permission denied for tool '{tool_name}'")
        if decision == PermissionDecision.REQUIRES_APPROVAL:
            return AgentRunResult(status="requires_approval", message=f"Tool '{tool_name}' requires explicit approval")

        tool_def = get_tool(tool_name)  # raises KeyError for a truly unknown name
        method = getattr(self.context.tools, tool_name)

        result: ToolResult = method(approved=approved, **kwargs) if tool_def.mutating else method(**kwargs)
        self.context.history.append({"tool": tool_name, "kwargs": kwargs, "status": result.status.value})
        return AgentRunResult(status="ok" if result.status == ToolStatus.OK else result.status.value, message=result.message, tool_result=result)

    def run_autonomous_task(self, user_task: str) -> AgentRunResult:
        """The full model -> planner -> tool -> observation -> continuation
        loop. Not implemented - see module docstring."""
        return AgentRunResult(
            status=KODRA_AGENT_RUNTIME_STATUS,
            message=(
                "Kodra Agent has no autonomous runtime loop yet. Individual "
                "tools can be invoked directly via call_tool(), but there is "
                "no planner turning model output into tool calls."
            ),
        )
