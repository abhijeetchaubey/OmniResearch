from langgraph.graph import StateGraph, START, END
from backend.graph.state import ResearchState
from backend.graph.nodes import (
    planner_node,
    task_router_node,
    task_executor_node,
    literature_node,
)


def _should_continue(state: ResearchState) -> str:
    """Route to specialized agent or executor if a task was routed, otherwise end."""
    if state.get("status") == "task_routed":
        next_agent = state.get("next_agent")
        if next_agent == "literature_agent":
            return "literature_agent"
        return "task_executor"
    return END


def create_research_workflow():
    """
    Constructs and compiles the LangGraph research workflow.
    Workflow:
    START -> planner -> task_router
    task_router -> literature_agent -> task_router
    task_router -> task_executor -> task_router
    task_router -> END
    """
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("task_router", task_router_node)
    builder.add_node("literature_agent", literature_node)
    builder.add_node("task_executor", task_executor_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "task_router")
    builder.add_conditional_edges("task_router", _should_continue)
    builder.add_edge("literature_agent", "task_router")
    builder.add_edge("task_executor", "task_router")

    return builder.compile()


research_workflow = create_research_workflow()


