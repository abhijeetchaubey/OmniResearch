"""
Prompt templates for Literature Research Agent in OmniResearch.
Tailors query construction, empirical evidence extraction, and synthesis
to specific research tasks (Transformer, Traditional ML, Evaluation Methodology).
"""

QUERY_GENERATION_PROMPT = """You are a senior AI/ML research scientist.
Your goal is to formulate 3 targeted academic search queries for a scholarly literature search.

Research Question: {research_question}
Current Task Title: {task_title}
Task Description: {task_description}
Expected Output: {expected_output}

Generate 3 clean, keyword-focused search queries:
1. Primary query focusing on the specific model architectures or algorithms requested.
2. Secondary query focusing on the benchmark datasets, tasks, or empirical comparisons.
3. Focused query combining key algorithms with evaluation metrics or baseline benchmarks.

Rules:
- Do NOT include generic phrases like "Literature Review", "Survey of", or "Baseline Identification".
- Output exactly 3 queries, one per line, without numbering or extra commentary.
"""

EVIDENCE_EXTRACTION_PROMPT = """You are a research scientist extracting verifiable empirical facts from a scholarly paper abstract.

Paper Title: {title}
Abstract: {abstract}

Extract ONLY information explicitly stated in the abstract. Do NOT invent numbers, metrics, or datasets.
Return valid JSON with:
- "research_objective": Brief summary of the specific problem studied (1 sentence)
- "models_or_algorithms": List of model or algorithm names mentioned
- "datasets": List of dataset names mentioned
- "evaluation_metrics": List of evaluation metrics mentioned (e.g. Accuracy, F1, AUC)
- "reported_results": List of objects with {{"metric": string, "value": string, "dataset": string, "source_text": string}} for any explicit quantitative numbers in the text
- "key_findings": List of main empirical or methodological conclusions stated in text
- "limitations": List of stated limitations or "Full text not analyzed" if only abstract was available
"""

TASK_SYNTHESIS_PROMPT = """You are an expert research scientist writing a structured literature review section for:
Research Problem: {research_question}
Subtask: {task_title}
Task Focus: {task_description}

Retrieved Scholarly Papers and Extracted Evidence:
{papers_text}

Provide a 3-paragraph synthesized literature review:
1. Methodological Approaches: Detail the architectures, algorithms, or baseline configurations investigated in these works.
2. Empirical Benchmarks & Datasets: Summarize the datasets used, reported metrics, and comparative trade-offs.
3. Open Gaps & Relevance: Explain how these findings inform the next stage of this research plan.

Cite each paper using [Author et al., Year] format based strictly on the provided paper list.
"""
