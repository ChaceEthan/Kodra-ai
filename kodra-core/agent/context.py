"""
Shared state container passed through the (future) agent runtime loop.
Holds no logic of its own - just the workspace/model/permission state a
planner and tool executor both need to see.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent.permissions import PermissionPolicy
from agent.tools import KodraAgentTools


@dataclass
class AgentContext:
    workspace_root: str
    tools: KodraAgentTools
    permissions: PermissionPolicy
    trained_checkpoint_loaded: bool
    # Free-form transcript of AgentStep-like entries. Kept as plain dicts
    # here (rather than importing AgentStep) to avoid a circular import with
    # agent.tools; the runtime module is responsible for structuring these.
    history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def build(cls, workspace_root: str, trained_checkpoint_loaded: bool) -> "AgentContext":
        tools = KodraAgentTools(workspace_root)
        permissions = PermissionPolicy(
            require_approval=tools.require_approval,
            enable_terminal_tools=tools.enable_terminal_tools,
            trained_checkpoint_loaded=trained_checkpoint_loaded,
        )
        return cls(
            workspace_root=workspace_root,
            tools=tools,
            permissions=permissions,
            trained_checkpoint_loaded=trained_checkpoint_loaded,
        )
