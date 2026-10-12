import unittest
from unittest.mock import patch, MagicMock
import httpx
from fastapi.testclient import TestClient

from backend.main import app
from backend.agents.literature.agent import LiteratureAgent
from backend.agents.literature.schema import (
    PaperMetadata,
    ExtractedEvidence,
    LiteratureTaskResult
)
from backend.agents.literature.tools import (
    OpenAlexProvider,
    ArxivProvider,
    SemanticScholarProvider,
    normalize_doi,
    normalize_title,
    merge_paper_records,
    deduplicate_papers,
    compute_deterministic_relevance,
    filter_and_rank_papers,
    extract_evidence_from_paper,
    validate_task_success_criteria
)
from backend.graph.nodes import literature_node, task_router_node, task_executor_node
from backend.graph.workflow import research_workflow
from backend.graph.state import ResearchState


class TestOmniResearchLiteratureAgent(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_rq = "Compare Transformer-based models with traditional ML models for sentiment classification."

        # High quality sample papers
        self.transformer_paper = PaperMetadata(
            paper_id="https://doi.org/10.1145/transformer.sent",
            title="Transformer Models for Sentiment Classification: An Empirical Benchmark",
            authors=["Vaswani, A.", "Devlin, J."],
            abstract="We evaluate Transformer models against baselines. Our fine-tuned RoBERTa achieves 94.2% accuracy on the SST-2 dataset.",
            year=2024,
            url="https://doi.org/10.1145/transformer.sent",
            doi="10.1145/transformer.sent",
            source="openalex",
            citation_count=150
        )

        self.svm_paper = PaperMetadata(
            paper_id="https://doi.org/10.1145/svm.sent",
            title="Support Vector Machines and Naive Bayes for Sentiment Classification",
            authors=["Joachims, T.", "Pang, B."],
            abstract="We investigate traditional machine learning baselines including SVM and Naive Bayes on IMDb movie reviews.",
            year=2022,
            url="https://doi.org/10.1145/svm.sent",
            doi="10.1145/svm.sent",
            source="arxiv",
            citation_count=85
        )

        # Off-topic unrelated paper
        self.philosophy_paper = PaperMetadata(
            paper_id="https://doi.org/10.1017/philosophy.sent",
            title="The Theory of Moral Sentiments and Historical Ethics",
            authors=["Smith, A."],
            abstract="An inquiry into the nature of virtue, sympathy, and moral sentiments in classical ethics.",
            year=1759,
            url="https://doi.org/10.1017/philosophy.sent",
            doi="10.1017/philosophy.sent",
            source="openalex",
            citation_count=5000
        )

    # --------------------------------------------------------------------------
    # 1. Query Construction
    # --------------------------------------------------------------------------
    def test_query_construction_transformer_task(self):
        """1. Verify focused query formulation for Transformer task."""
        agent = LiteratureAgent()
        queries = agent.build_search_queries(
            task_title="Literature Survey on Transformer-based Sentiment Models",
            task_description="Search for Transformer models, fine-tuning strategies, and sentiment metrics.",
            research_question=self.sample_rq
        )
        self.assertTrue(len(queries) >= 1)
        self.assertNotIn("Literature Survey on Transformer-based Sentiment Models", queries)
        joined = " ".join(queries).lower()
        self.assertTrue("transformer" in joined or "sentiment" in joined)

    def test_query_construction_traditional_ml_task(self):
        """1b. Verify query formulation for Traditional ML task."""
        agent = LiteratureAgent()
        queries = agent.build_search_queries(
            task_title="Literature Survey: Traditional ML Baselines",
            task_description="Evaluate SVM, Naive Bayes, and Random Forest baselines for text classification.",
            research_question=self.sample_rq
        )
        joined = " ".join(queries).lower()
        self.assertTrue("support vector machine" in joined or "naive bayes" in joined or "traditional" in joined)

    # --------------------------------------------------------------------------
    # 2 & 3. Normalization (DOI, titles, malformed responses)
    # --------------------------------------------------------------------------
    def test_doi_normalization(self):
        """2 & 3. Verify robust DOI normalization from diverse URL formats."""
        self.assertEqual(normalize_doi("https://doi.org/10.1145/1234.5678"), "10.1145/1234.5678")
        self.assertEqual(normalize_doi("http://dx.doi.org/10.1016/j.patcog.2021.108"), "10.1016/j.patcog.2021.108")
        self.assertEqual(normalize_doi("doi:10.1007/978-3-030-58571-6_1"), "10.1007/978-3-030-58571-6_1")
        self.assertIsNone(normalize_doi("invalid_doi_format"))
        self.assertIsNone(normalize_doi(None))

    def test_openalex_abstract_reconstruction(self):
        """Verify OpenAlex inverted index correctly reconstructs full abstract text."""
        raw_item = {
            "display_name": "Test Paper",
            "abstract_inverted_index": {
                "Transformers": [0],
                "outperform": [1],
                "baselines": [2],
                "significantly.": [3]
            }
        }
        provider = OpenAlexProvider()
        abstract = provider._reconstruct_abstract(raw_item)
        self.assertEqual(abstract, "Transformers outperform baselines significantly.")

    # --------------------------------------------------------------------------
    # 4 & 5. Deduplication and Metadata Merging
    # --------------------------------------------------------------------------
    def test_deduplication_and_metadata_merging(self):
        """4 & 5. Verify DOI-based and title-based deduplication merges metadata."""
        paper_oa = PaperMetadata(
            paper_id="oa_1",
            title="Transformer Benchmarks",
            authors=["Vaswani, A."],
            abstract="",  # Missing abstract in OpenAlex
            year=2023,
            doi="10.1145/transformer.bench",
            source="openalex",
            citation_count=50
        )
        paper_arxiv = PaperMetadata(
            paper_id="arxiv_1",
            title="Transformer Benchmarks",
            authors=["Devlin, J."],  # Additional author in arXiv
            abstract="Full abstract present in arXiv record.",
            year=2023,
            doi="https://doi.org/10.1145/transformer.bench",
            source="arxiv",
            citation_count=30
        )

        deduped = deduplicate_papers([paper_oa, paper_arxiv])
        self.assertEqual(len(deduped), 1)
        merged = deduped[0]
        # Should merge authors
        self.assertIn("Vaswani, A.", merged.authors)
        self.assertIn("Devlin, J.", merged.authors)
        # Should keep non-empty abstract
        self.assertEqual(merged.abstract, "Full abstract present in arXiv record.")
        # Should keep max citation count
        self.assertEqual(merged.citation_count, 50)
        # Should combine source provenance
        self.assertEqual(merged.source, "openalex+arxiv")

    # --------------------------------------------------------------------------
    # 6. Relevance Ranking & Negative Domain Filtering
    # --------------------------------------------------------------------------
    def test_relevance_ranking_and_off_topic_filtering(self):
        """6. Verify off-topic papers (e.g. Adam Smith Moral Sentiments) are filtered or downgraded."""
        ranked = filter_and_rank_papers(
            papers=[self.philosophy_paper, self.transformer_paper, self.svm_paper],
            research_question=self.sample_rq,
            task_title="Transformer Sentiment Classification",
            task_description="Evaluate Transformer models on sentiment benchmarks",
            min_threshold=0.20
        )
        # The philosophy paper must be filtered out due to off-topic domain
        paper_titles = [p.title for p in ranked]
        self.assertNotIn(self.philosophy_paper.title, paper_titles)
        self.assertIn(self.transformer_paper.title, paper_titles)
        self.assertIn(self.svm_paper.title, paper_titles)
        # The transformer paper should rank first
        self.assertEqual(ranked[0].title, self.transformer_paper.title)
        self.assertGreater(ranked[0].relevance_score, 0.5)

    # --------------------------------------------------------------------------
    # 7. Missing Abstracts and Metadata Safety
    # --------------------------------------------------------------------------
    def test_missing_metadata_safety(self):
        """7. Verify missing fields (None year, empty abstract, empty authors) do not crash system."""
        sparse_paper = PaperMetadata(
            paper_id="sparse_1",
            title="A Sparse Paper Record",
            authors=[],
            abstract="",
            year=None,
            source="openalex"
        )
        ev = extract_evidence_from_paper(sparse_paper)
        self.assertEqual(ev.evidence_status, "unavailable")
        self.assertTrue(len(ev.limitations) > 0)
        self.assertIn("metadata-only", ev.limitations[0])

    # --------------------------------------------------------------------------
    # 8, 9 & 10. Provider Resilience (Timeouts, HTTP Errors, Rate Limits)
    # --------------------------------------------------------------------------
    def test_provider_resilience_partial_availability(self):
        """8 & 9. Verify agent succeeds when one provider times out or errors."""
        agent = LiteratureAgent()

        # Mock OpenAlex returning 1 paper, arXiv timing out, Semantic Scholar returning 429
        mock_oa_papers = [self.transformer_paper]
        mock_oa_status = {"provider": "openalex", "status": "ok", "retrieved": 1}
        mock_arxiv_status = {"provider": "arxiv", "status": "timeout", "retrieved": 0, "error": "Timeout"}
        mock_s2_status = {"provider": "semantic_scholar", "status": "rate_limited", "retrieved": 0, "error": "HTTP 429"}

        with patch.object(agent.openalex_provider, "search", return_value=(mock_oa_papers, mock_oa_status)), \
             patch.object(agent.arxiv_provider, "search", return_value=([], mock_arxiv_status)), \
             patch.object(agent.s2_provider, "search", return_value=([], mock_s2_status)):

            raw_papers, telemetry = agent.search_providers(["transformer sentiment"])
            self.assertEqual(len(raw_papers), 1)
            self.assertEqual(telemetry["openalex"]["status"], "ok")
            self.assertEqual(telemetry["arxiv"]["status"], "timeout")
            self.assertEqual(telemetry["semantic_scholar"]["status"], "rate_limited")

    # --------------------------------------------------------------------------
    # 11. Structured Evidence Extraction Without Fabrication
    # --------------------------------------------------------------------------
    def test_evidence_extraction_traceability(self):
        """11. Verify quantitative numbers are extracted ONLY when present in text, never fabricated."""
        ev = extract_evidence_from_paper(self.transformer_paper)
        self.assertIn("Transformer", ev.models_or_algorithms)
        self.assertIn("SST-2", ev.datasets)
        self.assertTrue(any(m.lower() == "accuracy" for m in ev.evaluation_metrics))
        self.assertTrue(len(ev.reported_results) > 0)
        self.assertEqual(ev.reported_results[0]["value"], "94.2%")
        self.assertIn(ev.reported_results[0]["metric"], ["accuracy", "Accuracy"])

        # Paper with no numbers in abstract
        paper_no_metrics = PaperMetadata(
            paper_id="test_no_num",
            title="Theoretical Review of Sentiment Analysis",
            authors=["Author, A."],
            abstract="We discuss the theoretical perspectives on sentiment classification algorithms.",
            year=2023,
            source="openalex"
        )
        ev_no_metrics = extract_evidence_from_paper(paper_no_metrics)
        self.assertEqual(len(ev_no_metrics.reported_results), 0, "No metrics must be invented when absent")

    # --------------------------------------------------------------------------
    # 12 & 13. Task Success Criteria Validation (Completed, Partial, Failed)
    # --------------------------------------------------------------------------
    def test_task_success_criteria_validation(self):
        """12 & 13. Verify task validation distinguishes completed, partial, and failed."""
        task_spec_high = {
            "task_id": "task_1",
            "title": "Literature Review",
            "success_criteria": "At least 5 relevant papers identified."
        }

        # Case A: 2 papers < target 5 -> status 'partial'
        evidence_list = [extract_evidence_from_paper(p) for p in [self.transformer_paper, self.svm_paper]]
        status_partial, summary_partial, warnings = validate_task_success_criteria(
            task_spec=task_spec_high,
            papers=[self.transformer_paper, self.svm_paper],
            evidence=evidence_list
        )
        self.assertEqual(status_partial, "partial")
        self.assertFalse(summary_partial["passed"])
        self.assertEqual(summary_partial["actual_count"], 2)
        self.assertEqual(summary_partial["target_count"], 5)

        # Case B: 0 papers -> status 'failed'
        status_failed, _, warnings_failed = validate_task_success_criteria(
            task_spec=task_spec_high,
            papers=[],
            evidence=[]
        )
        self.assertEqual(status_failed, "failed")

        # Case C: 2 papers with target 2 -> status 'completed'
        task_spec_target_2 = {"success_criteria": "At least 2 relevant papers identified."}
        status_completed, summary_completed, _ = validate_task_success_criteria(
            task_spec=task_spec_target_2,
            papers=[self.transformer_paper, self.svm_paper],
            evidence=evidence_list
        )
        self.assertEqual(status_completed, "completed")
        self.assertTrue(summary_completed["passed"])

    # --------------------------------------------------------------------------
    # 14, 15 & 16. State Updates and LangGraph Node Execution
    # --------------------------------------------------------------------------
    def test_literature_node_execution_and_paper_persistence(self):
        """14, 15 & 16. Verify literature_node updates state, dedupes papers, and updates completed_tasks."""
        state: ResearchState = {
            "research_id": "test_run_lit_node",
            "research_question": self.sample_rq,
            "current_task": "task_1",
            "completed_tasks": [],
            "failed_tasks": [],
            "papers": [],
            "retrieved_evidence": [],
            "task_results": {},
            "plan": {
                "subtasks": [
                    {
                        "task_id": "task_1",
                        "title": "Literature Survey on Transformer-based Sentiment Models",
                        "description": "Find relevant Transformer models.",
                        "category": "literature",
                        "assigned_agent": "literature_agent",
                        "success_criteria": "At least 1 relevant paper identified."
                    }
                ]
            }
        }

        # Mock search_providers to return high quality paper
        with patch.object(LiteratureAgent, "search_providers", return_value=([self.transformer_paper], {"openalex": {"status": "ok", "retrieved": 1}})):
            output = literature_node(state)

            self.assertEqual(output.get("status"), "task_completed")
            self.assertIn("task_1", output.get("completed_tasks", []))
            self.assertTrue(len(output.get("papers", [])) > 0)
            self.assertTrue(len(output.get("retrieved_evidence", [])) > 0)
            self.assertIn("task_1", output.get("task_results", {}))

            task_res = output["task_results"]["task_1"]
            self.assertEqual(task_res["status"], "completed")
            self.assertEqual(task_res["task_id"], "task_1")

    # --------------------------------------------------------------------------
    # 17. Overall Research Status When Other Agents Are Stubs
    # --------------------------------------------------------------------------
    def test_router_status_aggregation_with_stubs(self):
        """17. Verify overall status is 'completed_with_stubs' when placeholder agents ran."""
        state: ResearchState = {
            "plan": {
                "subtasks": [
                    {"task_id": "task_1", "dependencies": [], "assigned_agent": "literature_agent"},
                    {"task_id": "task_2", "dependencies": ["task_1"], "assigned_agent": "dataset_agent"}
                ]
            },
            "completed_tasks": ["task_1", "task_2"],
            "task_results": {
                "task_1": {"status": "completed"},
                "task_2": {"status": "not_implemented", "is_placeholder": True}
            }
        }
        res = task_router_node(state)
        self.assertEqual(res["status"], "completed_with_stubs")
        summary = res["execution_summary"]
        self.assertEqual(summary["stub_count"], 1)
        self.assertEqual(summary["completed_count"], 1)

    # --------------------------------------------------------------------------
    # 18. Backward Compatibility of the FastAPI Endpoint
    # --------------------------------------------------------------------------
    def test_api_post_research_plan_backward_compatibility(self):
        """18. Verify POST /research/plan API endpoint response shape is backward-compatible."""
        with patch.object(LiteratureAgent, "search_providers", return_value=([self.transformer_paper], {"openalex": {"status": "ok", "retrieved": 1}})):
            response = self.client.post(
                "/research/plan",
                json={"research_question": self.sample_rq}
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("research_id", data)
            self.assertIn("status", data)
            self.assertIn("plan", data)
            self.assertIn("papers", data)
            self.assertTrue(len(data["papers"]) > 0)
            first_p = data["papers"][0]
            self.assertIn("paper_id", first_p)
            self.assertIn("title", first_p)
            self.assertIn("source", first_p)


if __name__ == "__main__":
    unittest.main()
