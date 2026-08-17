import os
import sys
import tempfile
import unittest

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from agent.tool_registry import TOOL_REGISTRY, get_tool, list_tools
from agent.permissions import PermissionPolicy, PermissionDecision
from agent.context import AgentContext
from agent.runtime import AgentRuntime, KODRA_AGENT_RUNTIME_STATUS
from agent.planner import Planner


class TestToolRegistry(unittest.TestCase):
    def test_all_documented_tools_present(self):
        expected = {
            "read_file", "search_files", "search_code", "list_directory",
            "create_file", "edit_file", "apply_patch", "run_tests",
            "run_terminal", "git_status", "git_diff",
        }
        self.assertEqual(set(TOOL_REGISTRY.keys()), expected)

    def test_mutating_flags_correct(self):
        for name in ("read_file", "search_files", "search_code", "list_directory", "git_status", "git_diff"):
            self.assertFalse(get_tool(name).mutating, name)
        for name in ("create_file", "edit_file", "apply_patch", "run_tests", "run_terminal"):
            self.assertTrue(get_tool(name).mutating, name)

    def test_unknown_tool_raises(self):
        with self.assertRaises(KeyError):
            get_tool("delete_everything")

    def test_list_tools(self):
        self.assertEqual(len(list_tools()), len(TOOL_REGISTRY))


class TestPermissionPolicy(unittest.TestCase):
    def test_read_only_always_allowed(self):
        policy = PermissionPolicy(require_approval=True, enable_terminal_tools=False, trained_checkpoint_loaded=False)
        self.assertEqual(policy.decide("read_file", approved=False, autonomous=True), PermissionDecision.ALLOW)

    def test_autonomous_mutation_denied_without_trained_checkpoint(self):
        policy = PermissionPolicy(require_approval=False, enable_terminal_tools=False, trained_checkpoint_loaded=False)
        # Even with require_approval=False and approved=True, autonomous
        # mutation is refused because there is no trained checkpoint.
        self.assertEqual(policy.decide("edit_file", approved=True, autonomous=True), PermissionDecision.DENY)

    def test_autonomous_mutation_allowed_with_trained_checkpoint_and_approval(self):
        policy = PermissionPolicy(require_approval=True, enable_terminal_tools=False, trained_checkpoint_loaded=True)
        self.assertEqual(policy.decide("edit_file", approved=False, autonomous=True), PermissionDecision.REQUIRES_APPROVAL)
        self.assertEqual(policy.decide("edit_file", approved=True, autonomous=True), PermissionDecision.ALLOW)

    def test_human_invoked_mutation_still_needs_approval(self):
        policy = PermissionPolicy(require_approval=True, enable_terminal_tools=False, trained_checkpoint_loaded=False)
        self.assertEqual(policy.decide("create_file", approved=False, autonomous=False), PermissionDecision.REQUIRES_APPROVAL)
        self.assertEqual(policy.decide("create_file", approved=True, autonomous=False), PermissionDecision.ALLOW)

    def test_terminal_disabled_denies_even_with_approval(self):
        policy = PermissionPolicy(require_approval=True, enable_terminal_tools=False, trained_checkpoint_loaded=True)
        self.assertEqual(policy.decide("run_terminal", approved=True, autonomous=False), PermissionDecision.DENY)

    def test_terminal_enabled_still_requires_approval_regardless_of_require_approval_flag(self):
        policy = PermissionPolicy(require_approval=False, enable_terminal_tools=True, trained_checkpoint_loaded=True)
        self.assertEqual(policy.decide("run_terminal", approved=False, autonomous=False), PermissionDecision.REQUIRES_APPROVAL)
        self.assertEqual(policy.decide("run_terminal", approved=True, autonomous=False), PermissionDecision.ALLOW)


class TestAgentRuntime(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        with open(os.path.join(self.workspace, "a.txt"), "w", encoding="utf-8") as f:
            f.write("hello\n")

    def test_read_only_tool_executes_for_real(self):
        context = AgentContext.build(self.workspace, trained_checkpoint_loaded=False)
        runtime = AgentRuntime(context)
        result = runtime.call_tool("read_file", relative_path="a.txt")
        self.assertEqual(result.status, "ok")
        self.assertIn("hello", result.tool_result.data)

    def test_mutating_tool_denied_when_autonomous_and_untrained(self):
        context = AgentContext.build(self.workspace, trained_checkpoint_loaded=False)
        runtime = AgentRuntime(context)
        result = runtime.call_tool("create_file", autonomous=True, approved=True, relative_path="x.py", content="x=1")
        self.assertEqual(result.status, "denied")
        self.assertFalse(os.path.exists(os.path.join(self.workspace, "x.py")))

    def test_mutating_tool_requires_approval_for_human_call(self):
        context = AgentContext.build(self.workspace, trained_checkpoint_loaded=False)
        runtime = AgentRuntime(context)
        result = runtime.call_tool("create_file", autonomous=False, relative_path="x.py", content="x=1")
        self.assertEqual(result.status, "requires_approval")

    def test_unknown_tool_returns_error_not_exception(self):
        context = AgentContext.build(self.workspace, trained_checkpoint_loaded=False)
        runtime = AgentRuntime(context)
        result = runtime.call_tool("nonexistent_tool")
        self.assertEqual(result.status, "error")

    def test_autonomous_loop_reports_not_yet_complete(self):
        context = AgentContext.build(self.workspace, trained_checkpoint_loaded=True)
        runtime = AgentRuntime(context)
        result = runtime.run_autonomous_task("fix the bug")
        self.assertEqual(result.status, KODRA_AGENT_RUNTIME_STATUS)
        self.assertEqual(result.status, "NOT_YET_COMPLETE")


class TestPlanner(unittest.TestCase):
    def test_plan_next_step_raises_not_implemented(self):
        planner = Planner()
        with self.assertRaises(NotImplementedError):
            planner.plan_next_step("some model output")


if __name__ == "__main__":
    unittest.main()
