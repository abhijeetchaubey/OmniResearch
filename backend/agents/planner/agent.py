import logging
from typing import Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from backend.config.settings import settings
from .schema import StructuredResearchPlan, SubTask
from .prompts import PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Research Planner Agent responsible for decomposing research questions
    into structured executable research plans.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.DEFAULT_MODEL_NAME

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", PLANNER_HUMAN_PROMPT_TEMPLATE),
        ])

    def _get_llm(self):
        if not self.api_key or self.api_key == "mock_key_for_testing":
            logger.warning("No valid GROQ_API_KEY provided. Planner running in fallback mode.")
            return None
        
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2,
            )
            return llm.with_structured_output(StructuredResearchPlan)
        except ImportError as ie:
            logger.warning(f"langchain_groq package is not installed in current python environment: {ie}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq LLM: {e}")
            return None

    def plan(self, research_question: str) -> StructuredResearchPlan:
        """
        Generate a structured research plan for a given research question.
        """
        llm_with_structure = self._get_llm()

        if llm_with_structure is None:
            return self._generate_fallback_plan(research_question, reason="LLM API key not configured or initialization failed.")

        try:
            formatted_prompt = self.prompt.format_messages(research_question=research_question)
            plan: StructuredResearchPlan = llm_with_structure.invoke(formatted_prompt)
            return plan
        except Exception as e:
            logger.exception(f"Error during PlannerAgent execution: {e}")
            return self._generate_fallback_plan(research_question, reason=str(e))

    def _generate_fallback_plan(self, research_question: str, reason: str) -> StructuredResearchPlan:
        """
        Generates a deterministic fallback plan when LLM call is unavailable or fails.
        """
        logger.info(f"Generating fallback plan for question: '{research_question}'. Reason: {reason}")
        return StructuredResearchPlan(
            title=f"Research Plan: {research_question[:60]}...",
            overview=f"Automated baseline plan generated for: '{research_question}'. (Note: {reason})",
            research_question=research_question,
            estimated_complexity="medium",
            subtasks=[
                SubTask(
                    task_id="task_1",
                    title="Literature Review & Baseline Identification",
                    description="Search arXiv, OpenAlex, and Semantic Scholar for relevant papers, benchmarks, and model architectures.",
                    category="literature",
                    assigned_agent="literature_agent",
                    required_tools=["arxiv_search", "openalex_search", "semantic_scholar_search"],
                    execution_type="parallel",
                    expected_output="Structured paper metadata list and theoretical background summary.",
                    dependencies=[],
                    priority="high",
                    success_criteria="At least 5 relevant papers identified with core methodologies and metrics extracted."
                ),
                SubTask(
                    task_id="task_2",
                    title="Dataset Discovery & Evaluation",
                    description="Identify suitable public datasets on Hugging Face / OpenML and evaluate dataset suitability.",
                    category="dataset",
                    assigned_agent="dataset_agent",
                    required_tools=["dataset_search", "huggingface_api"],
                    execution_type="sequential",
                    expected_output="Selected dataset metadata, schema specification, and data splits.",
                    dependencies=["task_1"],
                    priority="high",
                    success_criteria="Dataset selected, validated for schema compatibility, and train/val/test splits defined."
                ),
                SubTask(
                    task_id="task_3",
                    title="Experimental Setup & Code Execution",
                    description="Configure baseline model and proposed approach in sandbox environment with MLflow tracking.",
                    category="experiment",
                    assigned_agent="experiment_agent",
                    required_tools=["code_executor", "mlflow"],
                    execution_type="sequential",
                    expected_output="MLflow experiment run logs, model weights, and raw evaluation metrics.",
                    dependencies=["task_1", "task_2"],
                    priority="high",
                    success_criteria="Completed training and evaluation runs with logged parameters and metric outputs."
                ),
                SubTask(
                    task_id="task_4",
                    title="Results & Metrics Analysis",
                    description="Compute statistical comparison across model variants, analyze confusion matrices and performance metrics.",
                    category="analysis",
                    assigned_agent="analysis_agent",
                    required_tools=["metrics_analyzer", "visualization_reader"],
                    execution_type="sequential",
                    expected_output="Comparative metrics analysis tables and visualization plot artifacts.",
                    dependencies=["task_3"],
                    priority="medium",
                    success_criteria="Quantitative comparison tables and visual metric plots generated."
                ),
                SubTask(
                    task_id="task_5",
                    title="Verification & Citation Check",
                    description="Verify numerical accuracy of results, check source citations, and validate experimental claims.",
                    category="verification",
                    assigned_agent="verification_agent",
                    required_tools=["citation_checker", "numerical_checker"],
                    execution_type="sequential",
                    expected_output="Verification report matching claims to cited papers and experiment logs.",
                    dependencies=["task_4"],
                    priority="medium",
                    success_criteria="All claims matched with verified metric outputs and cited literature sources."
                ),
                SubTask(
                    task_id="task_6",
                    title="Research Report Generation",
                    description="Synthesize background, methodology, experimental findings, and discussion into structured research paper.",
                    category="writer",
                    assigned_agent="writer_agent",
                    required_tools=["report_renderer", "citation_formatter"],
                    execution_type="sequential",
                    expected_output="Final structured research report formatted in markdown with references.",
                    dependencies=["task_5"],
                    priority="high",
                    success_criteria="Complete markdown research report generated with references and results tables."
                ),
            ]
        )
