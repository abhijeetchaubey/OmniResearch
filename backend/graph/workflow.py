from langgraph.graph import StateGraph, START, END
from backend.graph.state import ResearchState
from backend.graph.nodes import planner_node, task_router_node


def create_research_workflow():
    """
    Constructs and compiles the LangGraph research workflow.
    Workflow sequence: START -> planner -> task_router -> END
    """
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("task_router", task_router_node)

    # Connect workflow graph edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "task_router")
    builder.set_finish_point("task_router")

    return builder.compile()


research_workflow = create_research_workflow()
