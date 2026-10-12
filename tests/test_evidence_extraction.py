"""
Regression tests for Literature Agent Evidence Extraction in OmniResearch.
Validates metric-value association, dataset extraction, evidence quality,
deduplication, and source validation using deterministic mock abstracts.
"""

import unittest
from backend.agents.literature.schema import PaperMetadata, ExtractedEvidence
from backend.agents.literature.tools import (
    extract_evidence_from_paper,
    extract_metric_results_from_text,
    extract_datasets_from_text,
    deduplicate_evidence,
    validate_extracted_evidence
)


class TestLiteratureEvidenceExtraction(unittest.TestCase):
    """
    Test suite verifying the 7 core requirements of Literature Agent evidence extraction.
    """

    def setUp(self):
        # Sample prompt abstract
        self.multimetric_abstract = (
            "The proposed model achieved 90.4% accuracy, 88% recall, 87.3% precision, "
            "and an F1-score of 87.7%."
        )

    # --------------------------------------------------------------------------
    # 1. Correct accuracy, precision, recall, and F1 extraction
    # --------------------------------------------------------------------------
    def test_correct_accuracy_precision_recall_f1_extraction(self):
        """1. Verify numerical values are associated with their exact canonical metrics."""
        paper = PaperMetadata(
            paper_id="paper_multi_metric_1",
            title="A Multi-Metric Empirical Study",
            abstract=self.multimetric_abstract,
            year=2024
        )
        ev = extract_evidence_from_paper(paper)

        expected_results = [
            {"metric": "accuracy", "value": "90.4%"},
            {"metric": "recall", "value": "88%"},
            {"metric": "precision", "value": "87.3%"},
            {"metric": "f1_score", "value": "87.7%"}
        ]

        self.assertEqual(len(ev.reported_results), 4)
        for expected in expected_results:
            match = next(
                (r for r in ev.reported_results if r["metric"] == expected["metric"] and r["value"] == expected["value"]),
                None
            )
            self.assertIsNotNone(match, f"Expected metric result {expected} was not found in {ev.reported_results}")

        # Check traceable connection
        for r in ev.reported_results:
            self.assertIn("source_text", r)
            self.assertIn(r["value"], r["source_text"])

    # --------------------------------------------------------------------------
    # 2. Incorrect metric-value associations
    # --------------------------------------------------------------------------
    def test_incorrect_metric_value_associations(self):
        """2. Verify metrics are NEVER assigned values reported for a different metric."""
        paper = PaperMetadata(
            paper_id="paper_swap_check",
            title="Boundary Association Verification",
            abstract=self.multimetric_abstract,
            year=2024
        )
        ev = extract_evidence_from_paper(paper)

        results_by_metric = {r["metric"]: r["value"] for r in ev.reported_results}

        # Check that metrics are strictly separated
        self.assertEqual(results_by_metric.get("accuracy"), "90.4%")
        self.assertNotEqual(results_by_metric.get("accuracy"), "88%")
        self.assertNotEqual(results_by_metric.get("accuracy"), "87.3%")
        self.assertNotEqual(results_by_metric.get("accuracy"), "87.7%")

        self.assertEqual(results_by_metric.get("recall"), "88%")
        self.assertNotEqual(results_by_metric.get("recall"), "90.4%")

        self.assertEqual(results_by_metric.get("precision"), "87.3%")
        self.assertNotEqual(results_by_metric.get("precision"), "90.4%")

        self.assertEqual(results_by_metric.get("f1_score"), "87.7%")
        self.assertNotEqual(results_by_metric.get("f1_score"), "90.4%")

        # Test additional metrics (AUC, BLEU, RMSE)
        other_abstract = "Our model reached 0.92 AUC, 34.5 BLEU, and achieved an RMSE of 0.15."
        paper_other = PaperMetadata(
            paper_id="paper_other_metrics",
            title="Additional Metrics Evaluation",
            abstract=other_abstract,
            year=2024
        )
        ev_other = extract_evidence_from_paper(paper_other)
        other_by_metric = {r["metric"]: r["value"] for r in ev_other.reported_results}
        self.assertEqual(other_by_metric.get("auc"), "0.92")
        self.assertEqual(other_by_metric.get("bleu"), "34.5")
        self.assertEqual(other_by_metric.get("rmse"), "0.15")

    # --------------------------------------------------------------------------
    # 3. Multiple datasets in one abstract
    # --------------------------------------------------------------------------
    def test_multiple_datasets_in_one_abstract(self):
        """3. Verify explicit dataset acronyms (ASTD, ASAD, TEAD) are preserved without inventing Twitter."""
        abstract = (
            "We analyze social sentiment. The proposed architecture was evaluated on "
            "ASTD, ASAD, and TEAD. Experiments show state-of-the-art results on all benchmarks."
        )
        paper = PaperMetadata(
            paper_id="paper_datasets_1",
            title="Multi-Dataset Arabic Sentiment Analysis",
            abstract=abstract,
            year=2023
        )
        ev = extract_evidence_from_paper(paper)

        self.assertIn("ASTD", ev.datasets)
        self.assertIn("ASAD", ev.datasets)
        self.assertIn("TEAD", ev.datasets)
        self.assertEqual(len(ev.datasets), 3)

        # Ensure Twitter was NOT invented simply because social sentiment was mentioned
        self.assertNotIn("Twitter", ev.datasets)
        self.assertNotIn("twitter", [d.lower() for d in ev.datasets])

    # --------------------------------------------------------------------------
    # 4. Missing metrics and datasets
    # --------------------------------------------------------------------------
    def test_missing_metrics_and_datasets(self):
        """4. Verify missing metrics and datasets return empty lists, not fabricated placeholders."""
        abstract = (
            "We present a conceptual discussion and qualitative framework for ethical AI. "
            "Our analysis highlights theoretical considerations and architectural paradigms."
        )
        paper = PaperMetadata(
            paper_id="paper_qualitative_1",
            title="Theoretical Perspectives on AI Systems",
            abstract=abstract,
            year=2023
        )
        ev = extract_evidence_from_paper(paper)

        self.assertEqual(ev.reported_results, [])
        self.assertEqual(ev.datasets, [])
        # Ensure no fabricated placeholder strings like "Reported Benchmark"
        for r in ev.reported_results:
            self.assertNotEqual(r.get("dataset"), "Reported Benchmark")

    # --------------------------------------------------------------------------
    # 5. Duplicate evidence records
    # --------------------------------------------------------------------------
    def test_duplicate_evidence_records(self):
        """5. Verify duplicate evidence is deduplicated via DOI or title+year, merging complementary metadata."""
        ev1 = ExtractedEvidence(
            paper_id="doi_10_1145_sample",
            title="Benchmarking Transformers",
            doi="10.1145/sample.bench",
            year=2024,
            research_objective="Investigates Transformers.",
            models_or_algorithms=["Transformer", "BERT"],
            datasets=["SST-2"],
            evaluation_metrics=["accuracy"],
            reported_results=[{"metric": "accuracy", "value": "93.4%", "source_text": "93.4% accuracy"}],
            key_findings=["Finding A"],
            limitations=["Limit 1"],
            evidence_source="abstract",
            evidence_status="partial"
        )

        ev2 = ExtractedEvidence(
            paper_id="arxiv_2401_001",
            title="Benchmarking Transformers",
            doi="https://doi.org/10.1145/sample.bench",
            year=2024,
            research_objective="Investigates Transformers comprehensively.",
            models_or_algorithms=["RoBERTa"],
            datasets=["IMDb"],
            evaluation_metrics=["f1_score"],
            reported_results=[{"metric": "f1_score", "value": "91.0%", "source_text": "91.0% F1"}],
            key_findings=["Finding B"],
            limitations=["Limit 2"],
            evidence_source="abstract",
            evidence_status="partial"
        )

        deduped = deduplicate_evidence([ev1, ev2])
        self.assertEqual(len(deduped), 1, "Duplicate evidence with matching DOI must be merged into 1 record")

        merged = deduped[0]
        # Complementary datasets and models must be merged
        self.assertIn("SST-2", merged.datasets)
        self.assertIn("IMDb", merged.datasets)
        self.assertIn("BERT", merged.models_or_algorithms)
        self.assertIn("RoBERTa", merged.models_or_algorithms)

        # Complementary reported results merged without duplication
        self.assertEqual(len(merged.reported_results), 2)
        metrics_merged = {r["metric"] for r in merged.reported_results}
        self.assertIn("accuracy", metrics_merged)
        self.assertIn("f1_score", metrics_merged)

        # Merged findings
        self.assertIn("Finding A", merged.key_findings)
        self.assertIn("Finding B", merged.key_findings)

    # --------------------------------------------------------------------------
    # 6. Abstract-only evidence being marked as partial
    # --------------------------------------------------------------------------
    def test_abstract_only_evidence_marked_as_partial(self):
        """6. Verify evidence from abstracts is marked partial and abstract provenance."""
        paper_abstract = PaperMetadata(
            paper_id="paper_with_abstract",
            title="Neural Sentiment Architectures",
            abstract="We evaluate sentiment classification models achieving 91% accuracy on SST-2.",
            year=2024
        )
        ev_abstract = extract_evidence_from_paper(paper_abstract)
        self.assertEqual(ev_abstract.evidence_status, "partial")
        self.assertEqual(ev_abstract.evidence_source, "abstract")

        paper_no_abstract = PaperMetadata(
            paper_id="paper_no_abstract",
            title="Metadata Record Only",
            abstract="",
            year=2023
        )
        ev_no_abstract = extract_evidence_from_paper(paper_no_abstract)
        self.assertEqual(ev_no_abstract.evidence_status, "unavailable")
        self.assertEqual(ev_no_abstract.evidence_source, "metadata")

    # --------------------------------------------------------------------------
    # 7. Results containing no unsupported numerical claims
    # --------------------------------------------------------------------------
    def test_results_containing_no_unsupported_numerical_claims(self):
        """7. Verify validation strips any numerical result or dataset not supported in the source text."""
        source_text = "The proposed model achieved 85.0% accuracy on SST-2."

        fabricated_evidence = ExtractedEvidence(
            paper_id="test_fabricated",
            title="Test Paper",
            abstract=source_text,
            datasets=["SST-2", "InventedFakeDataset"],
            reported_results=[
                {"metric": "accuracy", "value": "85.0%", "source_text": source_text},
                {"metric": "recall", "value": "99.9%", "source_text": "Unsupported 99.9% claim"}
            ],
            evidence_status="partial",
            evidence_source="abstract"
        )

        validated = validate_extracted_evidence(fabricated_evidence, source_text=source_text)

        # Supported accuracy 85.0% must remain
        self.assertEqual(len(validated.reported_results), 1)
        self.assertEqual(validated.reported_results[0]["value"], "85.0%")

        # Unsupported 99.9% recall must be stripped
        values = [r["value"] for r in validated.reported_results]
        self.assertNotIn("99.9%", values)

        # Unsupported dataset must be stripped
        self.assertIn("SST-2", validated.datasets)
        self.assertNotIn("InventedFakeDataset", validated.datasets)


if __name__ == "__main__":
    unittest.main()
