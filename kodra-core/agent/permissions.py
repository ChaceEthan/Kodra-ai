"""
Permission policy for the Kodra Agent tool loop.

This is the single place that decides whether a tool call may proceed. It
combines three independent gates, ALL of which must pass for a mutating
tool to execute:

  1. Runtime flags (KODRA_REQUIRE_TOOL_APPROVAL, KODRA_ENABLE_TERMINAL_TOOLS)
  2. Explicit human approval for this specific call (`approved=True`)
  3. Model readiness: a mutating tool call driven autonomously by the model
     (not by a human directly invoking it) is refused unless a TRAINED
     checkpoint is loaded. An untrained, randomly-initialized KodraGPT must
     never be allowed to autonomously edit files, run tests, or run shell
     commands - it has no learned judgment about whether doing so is safe
     or correct.
"""
from dataclasses import dataclass
from enum import Enum

from agent.tool_registry import get_tool


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    REQUIRES_APPROVAL = "requires_approval"
    DENY = "deny"


@dataclass
class PermissionPolicy:
    require_approval: bool
    enable_terminal_tools: bool
    trained_checkpoint_loaded: bool

    def decide(self, tool_name: str, approved: bool, autonomous: bool) -> PermissionDecision:
        """`autonomous=True` means the caller is the model/planner loop
        acting without a human directly in the loop for this call (the
        future case); `autonomous=False` means a human is the one invoking
        the tool right now (e.g. via a UI button), which is always allowed
        to proceed to the normal approval gate.

        Read-only tools are always ALLOW - they cannot mutate anything.
        """
        tool = get_tool(tool_name)

        if not tool.mutating:
            return PermissionDecision.ALLOW

        if tool.requires_terminal and not self.enable_terminal_tools:
            return PermissionDecision.DENY

        if autonomous and not self.trained_checkpoint_loaded:
            # Hard refusal - no approval flag can override this. An
            # untrained model has no basis for autonomous mutation.
            return PermissionDecision.DENY

        if tool.requires_terminal:
            # run_terminal always requires explicit approval, regardless of
            # require_approval - shell execution never skips confirmation.
            return PermissionDecision.ALLOW if approved else PermissionDecision.REQUIRES_APPROVAL

        if self.require_approval and not approved:
            return PermissionDecision.REQUIRES_APPROVAL

        return PermissionDecision.ALLOW
