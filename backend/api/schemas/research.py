from pydantic import BaseModel, Field
from backend.agents.planner.schema import StructuredResearchPlan


class ResearchPlanRequest(BaseModel):
    """
    Request model for generating a research plan.
    """
    research_question: str = Field(
        ...,
        min_length=5,
        description="High-level AI/ML research question or goal.",
        example="Compare Transformer-based models with traditional ML models for sentiment classification."
    )


class ResearchPlanResponse(BaseModel):
    """
    Response model for research plan endpoint.
    """
    research_id: str = Field(..., description="Unique identifier for the research run")
    status: str = Field(..., description="Status of the research workflow execution")
    plan: StructuredResearchPlan = Field(..., description="Generated structured research plan")
