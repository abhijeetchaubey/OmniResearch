import logging
from typing import Dict, Any, List

from backend.graph.state import ResearchState
from backend.agents.planner.agent import PlannerAgent
from backend.graph.routing import resolve_task_agent, RoutingError

logger = logging.getLogger(__name__)


def planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node for executing the Research Planner Agent.
    Transforms the research_question in state into a structured research plan.
    """
    research_question = state.get("research_question", "")
    if not research_question:
        logger.error("planner_node called without research_question in state.")
        return {
            "status": "error",
            "errors": state.get("errors", []) + [{"node": "planner", "error": "Missing research_question"}]
        }

    logger.info(f"Executing planner_node for research question: {research_question[:50]}...")
    agent = PlannerAgent()
    structured_plan = agent.plan(research_question)

    plan_dict = structured_plan.model_dump()
    first_task = plan_dict["subtasks"][0]["task_id"] if plan_dict.get("subtasks") else ""

    return {
        "plan": plan_dict,
        "status": "plan_generated",
        "current_task": first_task,
    }


def task_router_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node for deterministically routing the next executable task to its assigned specialized agent.
    
    Inspects plan subtasks and completed_tasks, finds the next executable subtask with satisfied
    dependencies, validates its assigned_agent using routing.py, and updates workflow state.
    """
    plan = state.get("plan")
    if not plan or not isinstance(plan, dict) or not plan.get("subtasks"):
        logger.error("task_router_node called without a valid plan in state.")
        existing_errors = state.get("errors", [])
        return {
            "status": "routing_error",
            "errors": existing_errors + [{"node": "task_router", "error": "Missing or empty research plan"}]
        }

    subtasks: List[Dict[str, Any]] = plan.get("subtasks", [])
    completed_tasks = set(state.get("completed_tasks", []))
    
    # Filter for subtasks that have not yet been completed
    uncompleted = [task for task in subtasks if task.get("task_id") not in completed_tasks]
    
    if not uncompleted:
        logger.info("All subtasks in research plan have been completed.")
        return {
            "status": "all_tasks_completed",
            "current_task": "",
            "next_agent": ""
        }

    # Find first subtask whose dependencies are all satisfied in completed_tasks
    executable_task: Dict[str, Any] = None
    for task in uncompleted:
        deps = task.get("dependencies", [])
        if all(dep in completed_tasks for dep in deps):
            executable_task = task
            break

    if not executable_task:
        logger.warning("No executable subtasks found; remaining subtasks have unfulfilled dependencies.")
        existing_errors = state.get("errors", [])
        return {
            "status": "routing_blocked",
            "errors": existing_errors + [{
                "node": "task_router",
                "error": "No executable tasks found; remaining tasks have unfulfilled dependencies"
            }]
        }

    # Resolve assigned agent via deterministic routing logic
    try:
        routed_info = resolve_task_agent(executable_task)
        task_id = executable_task.get("task_id")
        agent_name = routed_info["agent_name"]

        logger.info(f"task_router_node selected task '{task_id}' assigned to '{agent_name}'.")
        return {
            "current_task": task_id,
            "next_agent": agent_name,
            "status": "task_routed"
        }
    except RoutingError as err:
        logger.error(f"Routing error in task_router_node: {err}")
        existing_errors = state.get("errors", [])
        return {
            "status": "routing_error",
            "errors": existing_errors + [{
                "node": "task_router",
                "task_id": executable_task.get("task_id"),
                "error": str(err)
            }]
        }
