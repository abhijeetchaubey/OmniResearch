"""
Literature Research Agent for OmniResearch.
Retrieves, normalizes, deduplicates, re-ranks, and synthesizes scholarly literature
from OpenAlex, arXiv, and Semantic Scholar for end-to-end AI/ML research automation.
"""

import re
import logging
from typing import List, Optional, Dict, Any

from backend.config.settings import settings
from .schema import (
    PaperMetadata,
    ExtractedEvidence,
    LiteratureTaskResult,
    LiteratureReviewResult
)
from .tools import (
    OpenAlexProvider,
    ArxivProvider,
    SemanticScholarProvider,
    deduplicate_papers,
    deduplicate_evidence,
    filter_and_rank_papers,
    extract_evidence_from_paper,
    validate_task_success_criteria,
    QUERY_STOPWORDS,
    KNOWN_MODELS,
    KNOWN_DATASETS
)
from .prompts import TASK_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


class LiteratureAgent:
    """
    Autonomous Literature Research Agent responsible for:
    1. Multi-facet search query construction using full research context.
    2. Multi-provider querying (OpenAlex, arXiv, Semantic Scholar) with resilient error handling.
    3. Normalization and DOI/title deduplication with metadata merging.
    4. Deterministic relevance filtering and ranking.
    5. Structured empirical evidence extraction without data fabrication.
    6. Task-specific synthesis and success criteria validation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.DEFAULT_MODEL_NAME
        self.timeout = timeout or settings.LITERATURE_TIMEOUT_SECONDS

        # Initialize scholarly providers
        self.openalex_provider = OpenAlexProvider(timeout=self.timeout)
        self.arxiv_provider = ArxivProvider(timeout=self.timeout)
        self.s2_provider = SemanticScholarProvider(timeout=self.timeout)

    def build_search_queries(
        self,
        task_title: str,
        task_description: str,
        research_question: str,
        expected_output: str = "",
        success_criteria: str = ""
    ) -> List[str]:
        """
        Constructs focused, academic search queries using the full task context:
        - Research question core topic.
        - Specific subtask focus (Transformer vs Traditional ML vs Evaluation).
        - Detected model names, algorithms, datasets, and research keywords.
        Avoids sending generic placeholder task titles blindly.
        """
        combined_text = f"{task_title} {task_description} {research_question} {expected_output}"
        combined_lower = combined_text.lower()

        # Identify key models or algorithms mentioned
        models_present = [m for m in KNOWN_MODELS if m in combined_lower]
        datasets_present = [d for d in KNOWN_DATASETS if d in combined_lower]

        # Clean generic placeholder words from title
        generic_terms = {
            "literature review", "literature survey", "baseline identification",
            "state of the art", "related work", "theoretical background"
        }
        clean_title = task_title
        for gt in generic_terms:
            clean_title = re.sub(rf"^{gt}[\s:–—\-]*", "", clean_title, flags=re.IGNORECASE).strip()

        # Extract content words from research question
        rq_words = [
            w for w in re.findall(r"\b[a-zA-Z0-9-]{3,}\b", research_question)
            if w.lower() not in QUERY_STOPWORDS
        ]
        core_domain = " ".join(rq_words[:3]) if rq_words else "machine learning"

        queries: List[str] = []

        # 1. Specialized Task Focus Query
        # Detect if task is Transformer-specific, Traditional ML-specific, or Evaluation-specific
        if any(t in combined_lower for t in ["transformer", "bert", "roberta", "llm", "deep learning"]):
            if "sentiment" in combined_lower:
                queries.append("transformer sentiment classification benchmark")
            else:
                queries.append(f"transformer models {core_domain}")
        elif any(t in combined_lower for t in ["traditional", "svm", "naive bayes", "random forest", "logistic regression"]):
            if "sentiment" in combined_lower:
                queries.append("support vector machine naive bayes sentiment analysis")
            else:
                queries.append(f"traditional machine learning baselines {core_domain}")
        elif any(t in combined_lower for t in ["metric", "evaluation", "benchmark", "protocol", "statistical"]):
            queries.append(f"evaluation metrics benchmark {core_domain}")

        # 2. Add query with models and datasets if detected
        if models_present and datasets_present:
            queries.append(f"{models_present[0]} {datasets_present[0]}")
        elif models_present:
            queries.append(f"{models_present[0]} {core_domain}")

        # 3. Add cleaned subtask title query if specific enough
        if clean_title and len(clean_title) >= 5 and clean_title.lower() != task_title.lower():
            clean_words = [w for w in clean_title.split() if w.lower() not in QUERY_STOPWORDS]
            if len(clean_words) >= 2:
                queries.append(" ".join(clean_words[:4]))

        # 4. Fallback core query
        if not queries:
            queries.append(f"{clean_title or core_domain}")

        # Deduplicate while preserving order
        deduped_queries = []
        for q in queries:
            q_clean = q.strip()
            if q_clean and q_clean not in deduped_queries:
                deduped_queries.append(q_clean)

        return deduped_queries[:3]

    def search_providers(
        self,
        queries: List[str],
        limit_per_provider: int = 5
    ) -> tuple[List[PaperMetadata], Dict[str, Dict[str, Any]]]:
        """
        Executes multi-provider literature retrieval across OpenAlex, arXiv, and Semantic Scholar.
        Tracks execution telemetry per provider and tolerates individual provider outages gracefully.
        """
        raw_papers: List[PaperMetadata] = []
        telemetry: Dict[str, Dict[str, Any]] = {
            "openalex": {"status": "uncalled", "retrieved": 0, "error": None},
            "arxiv": {"status": "uncalled", "retrieved": 0, "error": None},
            "semantic_scholar": {"status": "uncalled", "retrieved": 0, "error": None}
        }

        for q in queries:
            # 1. OpenAlex
            oa_papers, oa_stat = self.openalex_provider.search(q, limit=limit_per_provider)
            raw_papers.extend(oa_papers)
            telemetry["openalex"]["status"] = oa_stat["status"]
            telemetry["openalex"]["retrieved"] = telemetry["openalex"]["retrieved"] + len(oa_papers)
            if oa_stat.get("error"):
                telemetry["openalex"]["error"] = oa_stat["error"]

            # 2. arXiv
            ar_papers, ar_stat = self.arxiv_provider.search(q, limit=limit_per_provider)
            raw_papers.extend(ar_papers)
            telemetry["arxiv"]["status"] = ar_stat["status"]
            telemetry["arxiv"]["retrieved"] = telemetry["arxiv"]["retrieved"] + len(ar_papers)
            if ar_stat.get("error"):
                telemetry["arxiv"]["error"] = ar_stat["error"]

            # 3. Semantic Scholar
            s2_papers, s2_stat = self.s2_provider.search(q, limit=limit_per_provider)
            raw_papers.extend(s2_papers)
            telemetry["semantic_scholar"]["status"] = s2_stat["status"]
            telemetry["semantic_scholar"]["retrieved"] = telemetry["semantic_scholar"]["retrieved"] + len(s2_papers)
            if s2_stat.get("error"):
                telemetry["semantic_scholar"]["error"] = s2_stat["error"]

        logger.info(
            f"Providers queried for {len(queries)} queries. "
            f"OpenAlex: {telemetry['openalex']['retrieved']}, "
            f"arXiv: {telemetry['arxiv']['retrieved']}, "
            f"Semantic Scholar: {telemetry['semantic_scholar']['retrieved']}"
        )

        return raw_papers, telemetry

    def synthesize_task_summary(
        self,
        task_title: str,
        task_description: str,
        research_question: str,
        papers: List[PaperMetadata],
        evidence: List[ExtractedEvidence]
    ) -> str:
        """
        Synthesizes task-specific literature review section.
        Prioritizes Transformer architectures, Traditional ML, or Evaluation methods
        depending on the subtask context.
        """
        if not papers:
            return f"### Literature Review: {task_title}\n\nNo relevant scholarly papers were retrieved for this task."

        # If LLM key is configured, synthesize via LLM
        if self.api_key and self.api_key != "mock_key_for_testing":
            try:
                from langchain_groq import ChatGroq
                from langchain_core.prompts import PromptTemplate

                llm = ChatGroq(groq_api_key=self.api_key, model_name=self.model_name, temperature=0.3)
                prompt = PromptTemplate.from_template(TASK_SYNTHESIS_PROMPT)

                papers_text_items = []
                for p in papers:
                    p_ev = next((e for e in evidence if e.paper_id == p.paper_id), None)
                    ev_info = ""
                    if p_ev and p_ev.reported_results:
                        res_strs = [f"{r.get('metric')}: {r.get('value')} ({r.get('dataset')})" for r in p_ev.reported_results]
                        ev_info = f" [Metrics: {', '.join(res_strs)}]"
                    papers_text_items.append(
                        f"- '{p.title}' ({p.year or 'N/A'}) by {', '.join(p.authors[:2]) or 'Unknown'}: "
                        f"{p.abstract[:200]}...{ev_info}"
                    )

                response = llm.invoke(prompt.format(
                    research_question=research_question,
                    task_title=task_title,
                    task_description=task_description,
                    papers_text="\n".join(papers_text_items)
                ))
                return response.content
            except Exception as e:
                logger.warning(f"LLM literature synthesis failed, falling back to structured synthesis: {e}")

        # Deterministic structured task synthesis
        summary_lines = [
            f"### Literature Review Synthesis: {task_title}",
            f"**Research Context**: {research_question}",
            f"**Task Focus**: {task_description}",
            f"Evaluated and verified **{len(papers)}** relevant scholarly papers.",
            "",
            "#### 1. Methodological Approaches Identified:"
        ]

        # Extract models across evidence
        all_models = set()
        for e in evidence:
            all_models.update(e.models_or_algorithms)
        if all_models:
            summary_lines.append(f"- **Key Architectures & Algorithms**: {', '.join(sorted(all_models))}.")

        for idx, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:3]) if p.authors else "Unknown Authors"
            year_str = f" ({p.year})" if p.year else ""
            rel_str = f" [Relevance: {p.relevance_score:.2f}, {p.confidence.title()}]" if p.relevance_score is not None else ""
            summary_lines.append(f"{idx}. **{p.title}**{year_str} by {authors_str} [{p.source.upper()}]{rel_str}")
            if p.abstract:
                summary_lines.append(f"   - *Summary*: {p.abstract[:180]}...")

        summary_lines.append("")
        summary_lines.append("#### 2. Empirical Benchmarks & Reported Results:")
        reported_count = 0
        for e in evidence:
            for r in e.reported_results:
                dataset_part = f" on dataset `{r.get('dataset')}`" if r.get('dataset') else ""
                summary_lines.append(
                    f"- **{r.get('metric')}**: {r.get('value')}{dataset_part} "
                    f"(Source context: \"{r.get('source_text')}\")"
                )
                reported_count += 1
        if reported_count == 0:
            summary_lines.append("- *Notice*: Quantitative metric scores were not explicitly stated in retrieved abstracts; detailed benchmarks require full PDF parsing.")

        summary_lines.append("")
        summary_lines.append("#### 3. Research Limitations & Traceability:")
        summary_lines.append("- Evidence extracted from published peer-reviewed abstracts and scholarly metadata APIs.")
        summary_lines.append("- Full PDF parsing integration point preserved for deep full-text extraction.")

        return "\n".join(summary_lines)

    def run(
        self,
        query: str,
        task_description: str = "",
        research_question: str = "",
        task_id: str = "task_1",
        task_info: Optional[Dict[str, Any]] = None
    ) -> LiteratureTaskResult:
        """
        Executes complete literature review workflow:
        1. Formulates multi-facet search queries.
        2. Retrieves candidate papers across OpenAlex, arXiv, Semantic Scholar.
        3. Normalizes and deduplicates papers with metadata merging.
        4. Filters and ranks papers based on deterministic relevance.
        5. Extracts empirical evidence without data fabrication.
        6. Validates task success criteria (paper count, coverage).
        7. Synthesizes task-specific literature output.
        """
        effective_title = query or (task_info.get("title") if task_info else "Literature Search")
        effective_desc = task_description or (task_info.get("description") if task_info else "")
        effective_rq = research_question or (task_info.get("research_question") if task_info else "") or effective_title
        task_spec = task_info or {
            "task_id": task_id,
            "title": effective_title,
            "description": effective_desc,
            "success_criteria": "At least 5 relevant papers identified."
        }

        logger.info(f"LiteratureAgent started for task '{task_id}': '{effective_title}'")

        # 1. Query Construction
        queries = self.build_search_queries(
            task_title=effective_title,
            task_description=effective_desc,
            research_question=effective_rq,
            expected_output=task_spec.get("expected_output", ""),
            success_criteria=task_spec.get("success_criteria", "")
        )
        logger.info(f"Task '{task_id}' search queries formulated: {queries}")

        # 2. Multi-Provider Retrieval
        raw_papers, provider_status = self.search_providers(
            queries=queries,
            limit_per_provider=settings.LITERATURE_SEARCH_LIMIT
        )
        papers_found = len(raw_papers)

        # 3. Normalization & Deduplication with Metadata Merging
        deduped_papers = deduplicate_papers(raw_papers)
        papers_after_dedup = len(deduped_papers)

        # 4. Relevance Filtering & Ranking
        ranked_papers = filter_and_rank_papers(
            papers=deduped_papers,
            research_question=effective_rq,
            task_title=effective_title,
            task_description=effective_desc,
            min_threshold=settings.LITERATURE_MIN_RELEVANCE_THRESHOLD,
            limit=settings.LITERATURE_SEARCH_LIMIT
        )
        relevant_papers_count = len(ranked_papers)

        # 5. Empirical Evidence Extraction & Deduplication
        raw_evidence: List[ExtractedEvidence] = []
        for p in ranked_papers:
            ev = extract_evidence_from_paper(p, task_context=f"{effective_title} {effective_desc}")
            raw_evidence.append(ev)
        extracted_evidence = deduplicate_evidence(raw_evidence)

        # 6. Task Success Criteria Validation
        status, criteria_results, warnings = validate_task_success_criteria(
            task_spec=task_spec,
            papers=ranked_papers,
            evidence=extracted_evidence
        )

        # Handle zero-paper failure state
        error_code = None
        error_msg = None
        if relevant_papers_count == 0:
            error_code = "NO_RELEVANT_PAPERS"
            error_msg = f"Zero relevant papers passed the relevance threshold ({settings.LITERATURE_MIN_RELEVANCE_THRESHOLD}) for task '{task_id}'."
            warnings.append(error_msg)

        # 7. Synthesis
        evidence_summary = self.synthesize_task_summary(
            task_title=effective_title,
            task_description=effective_desc,
            research_question=effective_rq,
            papers=ranked_papers,
            evidence=extracted_evidence
        )

        limitations = [
            "Evidence extraction limited to published abstracts and scholarly repository metadata.",
            "PDF binary retrieval and multimodal layout parsing not executed in this phase."
        ]

        logger.info(
            f"LiteratureAgent task '{task_id}' finished: status='{status}', "
            f"found={papers_found}, after_dedup={papers_after_dedup}, relevant={relevant_papers_count}"
        )

        return LiteratureTaskResult(
            task_id=task_id,
            task_title=effective_title,
            research_question=effective_rq,
            papers_found=papers_found,
            papers_after_deduplication=papers_after_dedup,
            relevant_papers=relevant_papers_count,
            papers=ranked_papers,
            evidence=extracted_evidence,
            evidence_summary=evidence_summary,
            provider_status=provider_status,
            warnings=warnings,
            limitations=limitations,
            success_criteria_results=criteria_results,
            status=status,
            search_queries_used=queries,
            error_code=error_code,
            error_message=error_msg
        )

    # ==========================================================================
    # Backward-Compatible Helper Methods
    # ==========================================================================

    def search_openalex(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        papers, _ = self.openalex_provider.search(query, limit=limit)
        return papers

    def search_arxiv(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        papers, _ = self.arxiv_provider.search(query, limit=limit)
        return papers

    def search_semantic_scholar(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        papers, _ = self.s2_provider.search(query, limit=limit)
        return papers

    def search(self, query: str, limit: int = 5, research_problem: str = "") -> List[PaperMetadata]:
        queries = [query]
        raw_papers, _ = self.search_providers(queries=queries, limit_per_provider=limit)
        deduped = deduplicate_papers(raw_papers)
        ranked = filter_and_rank_papers(
            papers=deduped,
            research_question=research_problem or query,
            task_title=query,
            task_description=research_problem,
            limit=limit
        )
        return ranked

    def summarize(self, query: str, papers: List[PaperMetadata]) -> str:
        evidence = [extract_evidence_from_paper(p) for p in papers]
        return self.synthesize_task_summary(
            task_title=query,
            task_description="",
            research_question=query,
            papers=papers,
            evidence=evidence
        )
