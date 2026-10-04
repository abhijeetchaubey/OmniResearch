import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.agents.planner.agent import PlannerAgent
from backend.agents.planner.schema import StructuredResearchPlan
from backend.graph.workflow import research_workflow
from backend.graph.state import ResearchState


class TestPlannerAgent(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_question = "Compare Transformer-based models with traditional ML models for sentiment classification."

    def test_planner_agent_direct(self):
        """Test PlannerAgent generates a valid StructuredResearchPlan with enhanced task schema."""
        agent = PlannerAgent()
        plan = agent.plan(self.sample_question)

        self.assertIsInstance(plan, StructuredResearchPlan)
        self.assertTrue(len(plan.subtasks) >= 4, "Plan should contain at least 4 subtasks")
        
        # Verify subtask categories and agent assignments
        task_categories = {subtask.category for subtask in plan.subtasks}
        assigned_agents = {subtask.assigned_agent for subtask in plan.subtasks}
        
        self.assertIn("literature", task_categories)
        self.assertIn("dataset", task_categories)
        self.assertIn("experiment", task_categories)
        
        self.assertIn("literature_agent", assigned_agents)
        self.assertIn("dataset_agent", assigned_agents)
        self.assertIn("experiment_agent", assigned_agents)

        for task in plan.subtasks:
            self.assertTrue(task.task_id, "Subtask must have task_id")
            self.assertTrue(task.assigned_agent, "Subtask must specify assigned_agent")
            self.assertIsInstance(task.required_tools, list, "Subtask must have list of required_tools")
            self.assertIn(task.execution_type, ["parallel", "sequential"], "Subtask execution_type must be parallel or sequential")
            self.assertTrue(task.expected_output, "Subtask must specify expected_output")
            self.assertTrue(task.success_criteria, "Subtask must have success_criteria")

    def test_langgraph_workflow(self):
        """Test LangGraph research workflow invocation."""
        initial_state: ResearchState = {
            "research_id": "test_123",
            "research_question": self.sample_question,
            "status": "pending"
        }

        final_state = research_workflow.invoke(initial_state)

        self.assertEqual(final_state.get("status"), "task_routed")
        self.assertIn("plan", final_state)
        self.assertIsNotNone(final_state["plan"].get("subtasks"))

    def test_post_research_plan_api(self):
        """Test POST /research/plan REST API endpoint returns enhanced task schema."""
        response = self.client.post(
            "/research/plan",
            json={"research_question": self.sample_question}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("research_id", data)
        self.assertEqual(data["status"], "task_routed")
        self.assertIn("plan", data)
        
        plan = data["plan"]
        self.assertIn("subtasks", plan)
        self.assertTrue(len(plan["subtasks"]) > 0)

        # Check first task contains all enhanced fields
        first_task = plan["subtasks"][0]
        self.assertIn("assigned_agent", first_task)
        self.assertIn("required_tools", first_task)
        self.assertIn("execution_type", first_task)
        self.assertIn("expected_output", first_task)


if __name__ == "__main__":
    unittest.main()
