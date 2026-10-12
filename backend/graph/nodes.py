import logging
from typing import Dict, Any, List

from backend.graph.state import ResearchState
from backend.agents.planner.agent import PlannerAgent
from backend.agents.literature.agent import LiteratureAgent
from backend.agents.literature.schema import PaperMetadata, ExtractedEvidence
from backend.agents.literature.tools import deduplicate_papers, deduplicate_evidence
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
    
    Inspects plan subtasks, completed_tasks, and failed_tasks. Finds the next executable subtask
    with satisfied dependencies, validates assigned_agent, and aggregates honest research status upon completion.
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
    failed_tasks = set(state.get("failed_tasks", []))
    
    # Filter for subtasks that have not yet been completed or failed
    uncompleted = [
        task for task in subtasks
        if task.get("task_id") not in completed_tasks and task.get("task_id") not in failed_tasks
    ]
    
    if not uncompleted:
        # All tasks in plan have been evaluated/executed. Aggregate honest research status.
        task_results = state.get("task_results", {})
        failed_list = list(failed_tasks)
        stub_tasks = [
            tid for tid, res in task_results.items()
            if isinstance(res, dict) and (res.get("status") == "not_implemented" or res.get("is_placeholder"))
        ]
        partial_tasks = [
            tid for tid, res in task_results.items()
            if isinstance(res, dict) and res.get("status") == "partial"
        ]
        fully_completed = [
            tid for tid, res in task_results.items()
            if isinstance(res, dict) and res.get("status") == "completed"
        ]

        summary = {
            "total_tasks": len(subtasks),
            "completed_count": len(fully_completed),
            "partial_count": len(partial_tasks),
            "stub_count": len(stub_tasks),
            "failed_count": len(failed_list)
        }

        if failed_list:
            final_status = "has_failed_tasks"
        elif stub_tasks or partial_tasks:
            final_status = "completed_with_stubs"
        else:
            final_status = "all_tasks_completed"

        logger.info(f"All subtasks processed. Final research status='{final_status}'. Summary: {summary}")
        return {
            "status": final_status,
            "current_task": "",
            "next_agent": "",
            "execution_summary": summary
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


def literature_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node for executing the Literature Research Agent.
    Finds, retrieves, normalizes, deduplicates, re-ranks, and synthesizes scholarly literature.
    Validates task success criteria and updates graph state.
    """
    current_task_id = state.get("current_task", "")
    research_question = state.get("research_question", "")

    # Retrieve subtask details from research plan
    plan = state.get("plan", {})
    subtasks = plan.get("subtasks", []) if isinstance(plan, dict) else []
    task_info = next((t for t in subtasks if t.get("task_id") == current_task_id), {})
    task_desc = task_info.get("description", research_question)
    task_title = task_info.get("title", research_question)

    logger.info(f"literature_node executing for task '{current_task_id}': {task_title[:50]}")
    agent = LiteratureAgent()
    task_result = agent.run(
        query=task_title,
        task_description=task_desc,
        research_question=research_question,
        task_id=current_task_id,
        task_info=task_info
    )

    # Accumulate task results
    task_results = dict(state.get("task_results") or {})
    task_results[current_task_id] = task_result.model_dump()

    # Accumulate and deduplicate papers across tasks using normalized DOI/title
    existing_papers_raw = list(state.get("papers") or [])
    existing_paper_objs = [PaperMetadata.model_validate(p) for p in existing_papers_raw if isinstance(p, dict)]
    all_paper_objs = existing_paper_objs + task_result.papers
    deduped_paper_objs = deduplicate_papers(all_paper_objs)
    serialized_papers = [p.model_dump() for p in deduped_paper_objs]

    # Accumulate and deduplicate evidence across tasks using DOI / normalized title+year
    existing_evidence_raw = list(state.get("retrieved_evidence") or [])
    existing_evidence_objs = [ExtractedEvidence.model_validate(e) for e in existing_evidence_raw if isinstance(e, dict)]
    all_evidence_objs = existing_evidence_objs + task_result.evidence
    deduped_evidence_objs = deduplicate_evidence(all_evidence_objs)
    serialized_evidence = [e.model_dump() for e in deduped_evidence_objs]

    # Update completed/failed tasks based on validated output
    completed = list(state.get("completed_tasks") or [])
    failed = list(state.get("failed_tasks") or [])
    errors = list(state.get("errors") or [])

    if task_result.status == "completed":
        if current_task_id not in completed:
            completed.append(current_task_id)
    elif task_result.status == "partial":
        # Useful output exists; permit downstream flow while registering partial status
        if current_task_id not in completed:
            completed.append(current_task_id)
        for w in task_result.warnings:
            errors.append({"node": "literature_agent", "task_id": current_task_id, "type": "warning", "message": w})
    else:  # failed
        if current_task_id not in failed:
            failed.append(current_task_id)
        errors.append({
            "node": "literature_agent",
            "task_id": current_task_id,
            "error_code": task_result.error_code or "TASK_FAILED",
            "error": task_result.error_message or "Task failed validation criteria"
        })

    return {
        "papers": serialized_papers,
        "retrieved_evidence": serialized_evidence,
        "task_results": task_results,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "errors": errors,
        "status": "task_completed"
    }


def task_executor_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executor node for specialized agents.
    Executes implemented agents or honestly records placeholder status for agents under development.
    """
    next_agent = state.get("next_agent", "")
    if next_agent == "literature_agent":
        return literature_node(state)

    current_task = state.get("current_task", "")
    completed = list(state.get("completed_tasks", []))
    task_results = dict(state.get("task_results") or {})

    # Explicitly register placeholder execution status without claiming real completion
    if current_task:
        task_results[current_task] = {
            "task_id": current_task,
            "assigned_agent": next_agent,
            "status": "not_implemented",
            "is_placeholder": True,
            "message": f"Agent '{next_agent}' is currently a placeholder stub and is not yet implemented."
        }
        if current_task not in completed:
            completed.append(current_task)
        logger.info(f"task_executor_node registered placeholder stub for task '{current_task}' (Agent: {next_agent}).")

    return {
        "completed_tasks": completed,
        "task_results": task_results,
        "status": "task_completed"
    }
