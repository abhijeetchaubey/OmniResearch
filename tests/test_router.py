import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.graph.registry import AGENT_REGISTRY, is_agent_registered, get_registered_agent
from backend.graph.routing import resolve_task_agent, RoutingError
from backend.graph.nodes import task_router_node
from backend.graph.workflow import research_workflow
from backend.graph.state import ResearchState


class TestTaskRouterAndRegistry(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_subtasks = [
            {
                "task_id": "task_1",
                "title": "Literature Search",
                "description": "Search arXiv and Semantic Scholar",
                "category": "literature",
                "assigned_agent": "literature_agent",
                "required_tools": ["arxiv_search"],
                "execution_type": "parallel",
                "expected_output": "Paper list",
                "dependencies": [],
                "priority": "high",
                "success_criteria": "5 papers found"
            },
            {
                "task_id": "task_2",
                "title": "Dataset Selection",
                "description": "Find sentiment dataset",
                "category": "dataset",
                "assigned_agent": "dataset_agent",
                "required_tools": ["dataset_search"],
                "execution_type": "sequential",
                "expected_output": "Selected dataset",
                "dependencies": ["task_1"],
                "priority": "high",
                "success_criteria": "Dataset loaded"
            },
            {
                "task_id": "task_3",
                "title": "Run Experiments",
                "description": "Train BERT baseline",
                "category": "experiment",
                "assigned_agent": "experiment_agent",
                "required_tools": ["code_executor"],
                "execution_type": "sequential",
                "expected_output": "MLflow run logs",
                "dependencies": ["task_1", "task_2"],
                "priority": "high",
                "success_criteria": "Training complete"
            }
        ]

        self.sample_plan = {
            "title": "Test Research Plan",
            "overview": "Overview of research plan",
            "research_question": "Test Question",
            "subtasks": self.sample_subtasks,
            "estimated_complexity": "medium"
        }

    def test_registry_validation(self):
        """Verify AGENT_REGISTRY contains all 6 required agents with tools."""
        required_agents = [
            "literature_agent",
            "dataset_agent",
            "experiment_agent",
            "analysis_agent",
            "verification_agent",
            "writer_agent"
        ]
        for agent_name in required_agents:
            self.assertTrue(is_agent_registered(agent_name), f"Agent '{agent_name}' must be registered.")
            spec = get_registered_agent(agent_name)
            self.assertIn("name", spec)
            self.assertIn("supported_tools", spec)
            self.assertTrue(len(spec["supported_tools"]) > 0, f"Agent '{agent_name}' must have supported_tools.")

    def test_valid_agent_routing(self):
        """Verify resolve_task_agent returns valid agent spec for subtask."""
        subtask = self.sample_subtasks[0]
        result = resolve_task_agent(subtask)
        self.assertEqual(result["agent_name"], "literature_agent")
        self.assertIn("supported_tools", result["spec"])

    def test_unknown_agent_handling(self):
        """Verify resolve_task_agent raises RoutingError for unregistered agent."""
        invalid_subtask = {
            "task_id": "task_99",
            "assigned_agent": "unknown_super_agent",
            "category": "literature"
        }
        with self.assertRaises(RoutingError):
            resolve_task_agent(invalid_subtask)

    def test_task_selection_with_satisfied_dependencies(self):
        """Verify router selects first task with satisfied dependencies when none completed."""
        state: ResearchState = {
            "plan": self.sample_plan,
            "completed_tasks": []
        }
        result = task_router_node(state)
        self.assertEqual(result["status"], "task_routed")
        self.assertEqual(result["current_task"], "task_1")
        self.assertEqual(result["next_agent"], "literature_agent")

    def test_skipping_tasks_with_incomplete_dependencies(self):
        """Verify router skips task_3 (which requires task_1 and task_2) if only task_1 is done."""
        state: ResearchState = {
            "plan": self.sample_plan,
            "completed_tasks": ["task_1"]
        }
        result = task_router_node(state)
        self.assertEqual(result["status"], "task_routed")
        self.assertEqual(result["current_task"], "task_2")
        self.assertEqual(result["next_agent"], "dataset_agent")

    def test_handling_already_completed_tasks(self):
        """Verify router skips already completed tasks (task_1, task_2) and routes task_3."""
        state: ResearchState = {
            "plan": self.sample_plan,
            "completed_tasks": ["task_1", "task_2"]
        }
        result = task_router_node(state)
        self.assertEqual(result["status"], "task_routed")
        self.assertEqual(result["current_task"], "task_3")
        self.assertEqual(result["next_agent"], "experiment_agent")

    def test_all_tasks_completed(self):
        """Verify router returns all_tasks_completed when all subtasks are finished."""
        state: ResearchState = {
            "plan": self.sample_plan,
            "completed_tasks": ["task_1", "task_2", "task_3"]
        }
        result = task_router_node(state)
        self.assertEqual(result["status"], "all_tasks_completed")
        self.assertEqual(result["current_task"], "")
        self.assertEqual(result["next_agent"], "")

    def test_complete_planner_to_router_workflow(self):
        """Verify end-to-end START -> planner -> task_router -> END flow."""
        initial_state: ResearchState = {
            "research_id": "res_test_routing_456",
            "research_question": "Compare Transformer-based models with traditional ML models for sentiment classification.",
            "completed_tasks": [],
            "status": "pending"
        }
        final_state = research_workflow.invoke(initial_state)

        self.assertEqual(final_state.get("status"), "task_routed")
        self.assertTrue(final_state.get("current_task").startswith("task_"))
        self.assertEqual(final_state.get("next_agent"), "literature_agent")


if __name__ == "__main__":
    unittest.main()
