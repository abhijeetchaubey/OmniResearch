"""
Centralized Agent Registry for OmniResearch workflow.
Defines all specialized agents and their supported tools/capabilities.
"""

from typing import Dict, Any


AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "literature_agent": {
        "name": "Literature Research Agent",
        "category": "literature",
        "description": "Finds, retrieves, parses, and summarizes scholarly literature and papers.",
        "supported_tools": [
            "arxiv_search",
            "openalex_search",
            "semantic_scholar_search",
            "pdf_parser",
            "citation_lookup",
            "tavily_search"
        ]
    },
    "dataset_agent": {
        "name": "Dataset Discovery Agent",
        "category": "dataset",
        "description": "Discovers, evaluates, and validates dataset suitability for research objectives.",
        "supported_tools": [
            "dataset_search",
            "huggingface_api",
            "openml_api",
            "kaggle_api",
            "dataset_metadata_parser"
        ]
    },
    "experiment_agent": {
        "name": "Experiment Agent",
        "category": "experiment",
        "description": "Generates ML code, runs experiments in isolated sandbox, and tracks metrics via MLflow.",
        "supported_tools": [
            "python_executor",
            "code_executor",
            "mlflow",
            "environment_setup"
        ]
    },
    "analysis_agent": {
        "name": "Analysis Agent",
        "category": "analysis",
        "description": "Interprets experiment metrics, generates statistical comparisons, and evaluates visual artifacts.",
        "supported_tools": [
            "pandas",
            "metrics_analyzer",
            "visualization_reader",
            "vlm_reader"
        ]
    },
    "verification_agent": {
        "name": "Verification Agent",
        "category": "verification",
        "description": "Verifies claims, citations, mathematical formulas, and numerical values against evidence.",
        "supported_tools": [
            "citation_checker",
            "claim_matcher",
            "numerical_checker",
            "experiment_validator"
        ]
    },
    "writer_agent": {
        "name": "Research Writer Agent",
        "category": "writer",
        "description": "Synthesizes verified findings into a structured academic research paper.",
        "supported_tools": [
            "report_renderer",
            "citation_formatter",
            "markdown_exporter"
        ]
    }
}


def is_agent_registered(agent_name: str) -> bool:
    """Check if an agent name exists in AGENT_REGISTRY."""
    return agent_name in AGENT_REGISTRY


def get_registered_agent(agent_name: str) -> Dict[str, Any]:
    """
    Retrieve agent specification from AGENT_REGISTRY.
    Raises KeyError if agent is not registered.
    """
    if not is_agent_registered(agent_name):
        raise KeyError(f"Agent '{agent_name}' is not registered in AGENT_REGISTRY.")
    return AGENT_REGISTRY[agent_name]
