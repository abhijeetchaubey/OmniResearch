"""
Shared LangGraph workflow state for OmniResearch agent system.
Communicates state between all specialized agents in the workflow graph.
"""

from typing import List, Dict, Any, TypedDict, Optional


class ResearchState(TypedDict, total=False):
    """
    Shared LangGraph workflow state for OmniResearch agent system.
    Communicates state between all specialized agents in the workflow graph.
    """
    research_id: str
    research_question: str
    plan: Dict[str, Any]             # Serialized StructuredResearchPlan dict
    current_task: str
    next_agent: str                  # Name of the registered agent assigned to execute current_task
    completed_tasks: List[str]
    failed_tasks: List[str]          # List of task_ids that failed execution
    task_results: Dict[str, Any]     # Map of task_id -> task execution outputs
    papers: List[Dict[str, Any]]
    datasets: List[Dict[str, Any]]
    retrieved_evidence: List[Dict[str, Any]]
    experiment_plan: Dict[str, Any]
    experiment_runs: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    verification: Dict[str, Any]
    errors: List[Dict[str, Any]]
    report: str
    status: str
