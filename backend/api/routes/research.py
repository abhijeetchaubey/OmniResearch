import uuid
import logging
from fastapi import APIRouter, HTTPException, status

from backend.api.schemas.research import ResearchPlanRequest, ResearchPlanResponse
from backend.agents.planner.schema import StructuredResearchPlan
from backend.graph.workflow import research_workflow
from backend.graph.state import ResearchState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research"])


@router.post(
    "/plan",
    response_model=ResearchPlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a structured research plan",
    description="Accepts a high-level research question and runs the Planner Agent via LangGraph to generate an executable research plan."
)
async def generate_research_plan(payload: ResearchPlanRequest) -> ResearchPlanResponse:
    research_id = f"res_{uuid.uuid4().hex[:12]}"
    
    initial_state: ResearchState = {
        "research_id": research_id,
        "research_question": payload.research_question,
        "completed_tasks": [],
        "papers": [],
        "datasets": [],
        "retrieved_evidence": [],
        "experiment_runs": [],
        "errors": [],
        "status": "pending"
    }

    try:
        final_state = research_workflow.invoke(initial_state)
        
        plan_dict = final_state.get("plan")
        if not plan_dict:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate plan from research workflow."
            )

        structured_plan = StructuredResearchPlan.model_validate(plan_dict)

        return ResearchPlanResponse(
            research_id=research_id,
            status=final_state.get("status", "completed"),
            plan=structured_plan
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error handling /research/plan request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while planning: {str(e)}"
        )
