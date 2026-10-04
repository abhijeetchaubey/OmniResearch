from typing import List, Literal
from pydantic import BaseModel, Field


TaskCategory = Literal[
    "literature",
    "dataset",
    "experiment",
    "analysis",
    "verification",
    "writer"
]

AssignedAgent = Literal[
    "literature_agent",
    "dataset_agent",
    "experiment_agent",
    "analysis_agent",
    "verification_agent",
    "writer_agent"
]

ExecutionType = Literal["parallel", "sequential"]

TaskPriority = Literal["high", "medium", "low"]


class SubTask(BaseModel):
    """
    Structured definition of a single subtask in the research workflow.
    """
    task_id: str = Field(
        ...,
        description="Unique identifier for the subtask (e.g., task_1, task_2)"
    )
    title: str = Field(
        ...,
        description="Short summary title of the task"
    )
    description: str = Field(
        ...,
        description="Detailed description of what needs to be accomplished"
    )
    category: TaskCategory = Field(
        ...,
        description="Category of the task"
    )
    assigned_agent: AssignedAgent = Field(
        ...,
        description="Specialized agent assigned to execute this task (e.g., literature_agent, dataset_agent, experiment_agent, analysis_agent, verification_agent, writer_agent)"
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="List of tools/APIs required by the agent for this task (e.g., arxiv_search, dataset_search, code_executor, mlflow)"
    )
    execution_type: ExecutionType = Field(
        default="sequential",
        description="Execution mode: parallel (can run concurrently with independent tasks) or sequential (must run after dependencies complete)"
    )
    expected_output: str = Field(
        ...,
        description="Description of the structured artifact or output expected from this task"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task_ids that must be completed before this task can start"
    )
    priority: TaskPriority = Field(
        default="medium",
        description="Priority level of the task"
    )
    success_criteria: str = Field(
        ...,
        description="Concrete, verifiable condition to mark task as completed"
    )


class StructuredResearchPlan(BaseModel):
    """
    Executable research plan decomposed into structured subtasks.
    """
    title: str = Field(
        ...,
        description="Title of the research project based on user question"
    )
    overview: str = Field(
        ...,
        description="Executive summary of the research methodology and goals"
    )
    research_question: str = Field(
        ...,
        description="Original research question submitted by user"
    )
    subtasks: List[SubTask] = Field(
        ...,
        description="Ordered list of executable subtasks with explicit dependencies"
    )
    estimated_complexity: Literal["low", "medium", "high", "very_high"] = Field(
        default="medium",
        description="Estimated overall research complexity"
    )
