"""
Data schemas for Literature Research Agent in OmniResearch.
Defines normalized scholarly paper records, extracted empirical evidence,
provider statuses, and task-specific literature evaluation results.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """
    Normalized metadata for a scholarly paper across academic repositories.
    """
    paper_id: str = Field(..., description="Unique paper identifier (DOI, arXiv ID, or URL)")
    title: str = Field(..., description="Normalized paper title")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(default="", description="Paper abstract or summary")
    year: Optional[int] = Field(default=None, description="Publication year")
    url: Optional[str] = Field(default=None, description="Direct URL or landing page link")
    doi: Optional[str] = Field(default=None, description="Normalized Digital Object Identifier (e.g. 10.1145/...)")
    source: str = Field(default="openalex", description="Source repository provenance (openalex, arxiv, semantic_scholar)")
    citation_count: int = Field(default=0, description="Reported citation count")
    relevance_score: Optional[float] = Field(default=None, description="Calculated relevance score (0.0 to 1.0) relative to research question")
    matched_query: Optional[str] = Field(default=None, description="Search query that retrieved this record")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of retrieval"
    )
    relevance_explanation: Optional[str] = Field(default=None, description="Rationale for computed relevance score")
    confidence: str = Field(default="medium", description="Confidence category: high, medium, borderline, filtered")


class ExtractedEvidence(BaseModel):
    """
    Empirical research evidence extracted from a retrieved scholarly paper.
    Distinguishes strictly between facts extracted from text and unavailable data.
    """
    paper_id: str = Field(..., description="Identifier of the source paper")
    title: Optional[str] = Field(default=None, description="Paper title for deduplication")
    doi: Optional[str] = Field(default=None, description="Paper DOI for deduplication")
    year: Optional[int] = Field(default=None, description="Publication year")
    research_objective: str = Field(default="", description="Core problem or research objective addressed")
    models_or_algorithms: List[str] = Field(default_factory=list, description="Machine learning models or algorithms evaluated")
    datasets: List[str] = Field(default_factory=list, description="Benchmark or empirical datasets used")
    evaluation_metrics: List[str] = Field(default_factory=list, description="Evaluation metrics measured (e.g., Accuracy, F1, AUC)")
    reported_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Verifiable quantitative metric outcomes extracted from text (metric, value, dataset, source_text)"
    )
    key_findings: List[str] = Field(default_factory=list, description="Substantive qualitative or methodological conclusions")
    limitations: List[str] = Field(default_factory=list, description="Reported constraints, scope boundaries, or missing full text")
    evidence_source: str = Field(default="abstract", description="Provenance of evidence: abstract, full_text, metadata, llm_inferred")
    evidence_status: str = Field(default="partial", description="Status of evidence: extracted, partial, unavailable")


class LiteratureTaskResult(BaseModel):
    """
    Comprehensive structured outcome of a specialized literature research task.
    Includes validation against requested success criteria and provider execution telemetry.
    """
    task_id: str = Field(..., description="Subtask identifier from research plan")
    task_title: str = Field(..., description="Title of the executed subtask")
    research_question: str = Field(..., description="Overarching research problem")
    papers_found: int = Field(default=0, description="Raw count of candidate papers retrieved across all providers")
    papers_after_deduplication: int = Field(default=0, description="Count of papers after deduplication and metadata merging")
    relevant_papers: int = Field(default=0, description="Count of papers passing relevance threshold")
    papers: List[PaperMetadata] = Field(default_factory=list, description="Top ranked, relevant, deduplicated papers")
    evidence: List[ExtractedEvidence] = Field(default_factory=list, description="Structured evidence extracted from relevant papers")
    evidence_summary: str = Field(default="", description="Synthesized literature review summary for this specific task")
    provider_status: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Telemetry per provider (status, records_returned, latency_ms, errors)"
    )
    warnings: List[str] = Field(default_factory=list, description="Execution warnings or missing coverage notices")
    limitations: List[str] = Field(default_factory=list, description="Known empirical limitations (e.g., abstract-only analysis)")
    success_criteria_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation against subtask success criteria (target vs actual, passed, details)"
    )
    status: str = Field(default="completed", description="Task validation status: completed, partial, failed")
    search_queries_used: List[str] = Field(default_factory=list, description="Targeted queries executed across providers")
    error_code: Optional[str] = Field(default=None, description="Error code if retrieval encountered fatal errors")
    error_message: Optional[str] = Field(default=None, description="Detailed explanation of error encountered")

    # Backward-compatible properties
    @property
    def query(self) -> str:
        return self.task_title

    @property
    def summary(self) -> str:
        return self.evidence_summary

    @property
    def total_found(self) -> int:
        return len(self.papers)


# Backward-compatible alias for existing imports
LiteratureReviewResult = LiteratureTaskResult
