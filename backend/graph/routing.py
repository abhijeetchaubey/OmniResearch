"""
Deterministic task-routing logic for OmniResearch workflow graph.
Resolves subtasks to registered specialized agents.
"""

import logging
from typing import Dict, Any

from backend.graph.registry import is_agent_registered, get_registered_agent

logger = logging.getLogger(__name__)


class RoutingError(Exception):
    """Exception raised when task routing fails."""
    pass


def resolve_task_agent(subtask: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic task-routing logic that inspects a subtask's assigned_agent,
    validates that the agent exists in AGENT_REGISTRY, and returns the agent specification.
    
    No LLM calls are used here as the Planner has already assigned the agent.
    """
    task_id = subtask.get("task_id", "unknown_task")
    assigned_agent = subtask.get("assigned_agent")

    # Fallback to category-based agent resolution if assigned_agent is missing
    if not assigned_agent:
        category = subtask.get("category")
        if category:
            candidate_agent = f"{category}_agent"
            if is_agent_registered(candidate_agent):
                assigned_agent = candidate_agent

    if not assigned_agent:
        raise RoutingError(f"Subtask '{task_id}' is missing 'assigned_agent' field.")

    if not is_agent_registered(assigned_agent):
        raise RoutingError(f"Assigned agent '{assigned_agent}' for task '{task_id}' is not registered in AGENT_REGISTRY.")

    agent_spec = get_registered_agent(assigned_agent)
    logger.info(f"Subtask '{task_id}' successfully routed to agent '{assigned_agent}'.")
    return {
        "agent_name": assigned_agent,
        "spec": agent_spec
    }
