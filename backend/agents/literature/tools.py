"""
Specialized tool modules for Literature Research Agent in OmniResearch.
Includes scholarly repository providers (OpenAlex, arXiv, Semantic Scholar),
normalization and deduplication with metadata merging, deterministic relevance ranking,
empirical evidence extraction, and task success criteria validation.
"""

import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import xml.etree.ElementTree as ET

import httpx

from backend.config.settings import settings
from .schema import PaperMetadata, ExtractedEvidence

logger = logging.getLogger(__name__)

# Irrelevant domain keywords to filter out obvious false-positives
OFF_TOPIC_DOMAINS = {
    "philosophy": ["moral sentiments", "theology", "ethics of", "virtue", "epistemology", "metaphysics", "theological"],
    "clinical_surgery": ["cardiac arrest", "myocardial", "surgical resection", "oncology", "chemotherapy", "pediatric surgery"],
    "botany_agriculture": ["crop yield", "fertilizer", "soil salinity", "botanical extract", "agronomy"]
}

# Standard stop words for academic NLP query formulation
QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "with", "by", "to", "at",
    "from", "into", "through", "during", "before", "after", "above", "below", "between",
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "compare", "comparing", "comparison", "based", "using", "investigate",
    "investigating", "study", "studies", "task", "subtask", "identification", "review", "survey",
    "overview", "baseline", "paper", "papers", "literature", "state-of-the-art"
}

# Known AI/ML model terms for facet matching
KNOWN_MODELS = [
    "transformer", "bert", "roberta", "distilbert", "deberta", "albert", "xlnet", "electra",
    "gpt", "t5", "bart", "llama", "mistral", "svm", "support vector machine", "naive bayes",
    "logistic regression", "random forest", "decision tree", "gradient boosting", "xgboost",
    "lightgbm", "catboost", "lstm", "bilstm", "gru", "cnn", "word2vec", "fasttext", "tfidf", "tf-idf"
]

# Known NLP & ML benchmark datasets
KNOWN_DATASETS = [
    "imdb", "sst-2", "sst-5", "sst", "yelp", "amazon reviews", "sentiment140", "twitter",
    "rotten tomatoes", "glue", "superglue", "squad", "mnli", "qnli", "mrpc", "qqp", "ag news",
    "financial phrasebank", "semeval", "multinli"
]

# Known evaluation metrics
KNOWN_METRICS = [
    "accuracy", "macro-f1", "micro-f1", "weighted-f1", "f1-score", "f1 score", "f1",
    "precision", "recall", "roc-auc", "roc auc", "auc", "matthews correlation", "mcc",
    "cross-entropy", "perplexity", "mean squared error", "mse", "latency", "inference time",
    "parameters", "flops"
]


# ==============================================================================
# 1. PROVIDERS (OpenAlex, arXiv, Semantic Scholar)
# ==============================================================================

class OpenAlexProvider:
    """Scholarly search provider for OpenAlex works API."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self.api_key = api_key or settings.OPENALEX_API_KEY
        self.timeout = timeout
        self.headers = {"User-Agent": "OmniResearch/0.1.0 (mailto:researcher@omniresearch.ai)"}
        if self.api_key:
            self.headers["api_key"] = self.api_key

    def _reconstruct_abstract(self, item: Dict[str, Any]) -> str:
        inv = item.get("abstract_inverted_index")
        if isinstance(inv, dict) and inv:
            try:
                word_positions = []
                for word, positions in inv.items():
                    if isinstance(positions, list):
                        for pos in positions:
                            word_positions.append((pos, word))
                word_positions.sort(key=lambda x: x[0])
                return " ".join([w for _, w in word_positions])
            except Exception:
                pass
        return item.get("abstract") or ""

    def search(self, query: str, limit: int = 5) -> Tuple[List[PaperMetadata], Dict[str, Any]]:
        url = "https://api.openalex.org/works"
        params = {"search": query, "per-page": limit}
        papers: List[PaperMetadata] = []
        status: Dict[str, Any] = {"provider": "openalex", "status": "ok", "retrieved": 0, "error": None}

        start_time = time.time()
        for attempt in range(settings.LITERATURE_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                    res = client.get(url, params=params)
                    if res.status_code == 200:
                        data = res.json()
                        for item in data.get("results", []):
                            title = item.get("display_name") or ""
                            if not title:
                                continue
                            authors = [
                                a.get("author", {}).get("display_name")
                                for a in item.get("authorships", [])
                                if a.get("author") and a.get("author", {}).get("display_name")
                            ]
                            doi_val = item.get("doi")
                            paper_id = doi_val or item.get("id") or f"openalex_{len(papers)}"
                            abstract = self._reconstruct_abstract(item)

                            papers.append(
                                PaperMetadata(
                                    paper_id=paper_id,
                                    title=title,
                                    authors=authors,
                                    abstract=abstract,
                                    year=item.get("publication_year"),
                                    url=doi_val or item.get("id"),
                                    doi=doi_val,
                                    source="openalex",
                                    citation_count=item.get("cited_by_count", 0),
                                    matched_query=query
                                )
                            )
                        status["retrieved"] = len(papers)
                        status["latency_ms"] = int((time.time() - start_time) * 1000)
                        return papers, status
                    elif res.status_code in (429, 503):
                        if attempt < settings.LITERATURE_MAX_RETRIES:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        status["status"] = "rate_limited"
                        status["error"] = f"HTTP {res.status_code} Rate Limit"
                        break
                    else:
                        status["status"] = "error"
                        status["error"] = f"HTTP {res.status_code}"
                        break
            except httpx.TimeoutException:
                status["status"] = "timeout"
                status["error"] = "Request Timeout"
                logger.warning(f"OpenAlex timeout for query: '{query}'")
                break
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
                logger.warning(f"OpenAlex error: {e}")
                break

        status["latency_ms"] = int((time.time() - start_time) * 1000)
        return papers, status


class ArxivProvider:
    """Scholarly search provider for arXiv Atom Export API."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.headers = {"User-Agent": "OmniResearch/0.1.0 (mailto:researcher@omniresearch.ai)"}

    def search(self, query: str, limit: int = 5) -> Tuple[List[PaperMetadata], Dict[str, Any]]:
        url = "https://export.arxiv.org/api/query"
        clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
        params = {"search_query": f"all:{clean_q}", "start": "0", "max_results": str(limit)}
        papers: List[PaperMetadata] = []
        status: Dict[str, Any] = {"provider": "arxiv", "status": "ok", "retrieved": 0, "error": None}

        start_time = time.time()
        for attempt in range(settings.LITERATURE_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                    res = client.get(url, params=params)
                    if res.status_code == 200:
                        root = ET.fromstring(res.text)
                        ns = {"atom": "http://www.w3.org/2005/Atom"}
                        for entry in root.findall("atom:entry", ns):
                            title_el = entry.find("atom:title", ns)
                            summary_el = entry.find("atom:summary", ns)
                            id_el = entry.find("atom:id", ns)
                            if title_el is None or not title_el.text:
                                continue

                            authors = [
                                a.find("atom:name", ns).text
                                for a in entry.findall("atom:author", ns)
                                if a.find("atom:name", ns) is not None and a.find("atom:name", ns).text
                            ]
                            published_el = entry.find("atom:published", ns)
                            year = int(published_el.text[:4]) if published_el is not None and published_el.text else None
                            paper_url = id_el.text.strip() if id_el is not None and id_el.text else f"arxiv_{len(papers)}"

                            # Extract DOI if present in links or arxiv ID
                            doi_val = None
                            for link in entry.findall("atom:link", ns):
                                if link.attrib.get("title") == "doi":
                                    doi_val = link.attrib.get("href")

                            papers.append(
                                PaperMetadata(
                                    paper_id=paper_url,
                                    title=title_el.text.strip().replace("\n", " "),
                                    authors=authors,
                                    abstract=summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else "",
                                    year=year,
                                    url=paper_url,
                                    doi=doi_val,
                                    source="arxiv",
                                    citation_count=0,
                                    matched_query=query
                                )
                            )
                        status["retrieved"] = len(papers)
                        status["latency_ms"] = int((time.time() - start_time) * 1000)
                        return papers, status
                    elif res.status_code in (429, 503):
                        if attempt < settings.LITERATURE_MAX_RETRIES:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        status["status"] = "rate_limited"
                        status["error"] = f"HTTP {res.status_code}"
                        break
                    else:
                        status["status"] = "error"
                        status["error"] = f"HTTP {res.status_code}"
                        break
            except httpx.TimeoutException:
                status["status"] = "timeout"
                status["error"] = "Request Timeout"
                logger.warning(f"arXiv timeout for query: '{query}'")
                break
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
                logger.warning(f"arXiv query error: {e}")
                break

        status["latency_ms"] = int((time.time() - start_time) * 1000)
        return papers, status


class SemanticScholarProvider:
    """Scholarly search provider for Semantic Scholar Graph API."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self.api_key = api_key or settings.SEMANTIC_SCHOLAR_API_KEY
        self.timeout = timeout
        self.headers = {"User-Agent": "OmniResearch/0.1.0 (mailto:researcher@omniresearch.ai)"}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def search(self, query: str, limit: int = 5) -> Tuple[List[PaperMetadata], Dict[str, Any]]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,url,citationCount,externalIds"
        }
        papers: List[PaperMetadata] = []
        status: Dict[str, Any] = {"provider": "semantic_scholar", "status": "ok", "retrieved": 0, "error": None}

        start_time = time.time()
        for attempt in range(settings.LITERATURE_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                    res = client.get(url, params=params)
                    if res.status_code == 200:
                        data = res.json()
                        for item in data.get("data", []):
                            title = item.get("title")
                            if not title:
                                continue
                            authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
                            ext_ids = item.get("externalIds") or {}
                            doi_val = ext_ids.get("DOI")
                            paper_id = item.get("paperId") or doi_val or item.get("url") or f"s2_{len(papers)}"

                            papers.append(
                                PaperMetadata(
                                    paper_id=paper_id,
                                    title=title,
                                    authors=authors,
                                    abstract=item.get("abstract") or "",
                                    year=item.get("year"),
                                    url=item.get("url"),
                                    doi=doi_val,
                                    source="semantic_scholar",
                                    citation_count=item.get("citationCount") or 0,
                                    matched_query=query
                                )
                            )
                        status["retrieved"] = len(papers)
                        status["latency_ms"] = int((time.time() - start_time) * 1000)
                        return papers, status
                    elif res.status_code in (429, 503):
                        if attempt < settings.LITERATURE_MAX_RETRIES:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        status["status"] = "rate_limited"
                        status["error"] = f"HTTP {res.status_code}"
                        break
                    else:
                        status["status"] = "error"
                        status["error"] = f"HTTP {res.status_code}"
                        break
            except httpx.TimeoutException:
                status["status"] = "timeout"
                status["error"] = "Request Timeout"
                logger.warning(f"Semantic Scholar timeout for query: '{query}'")
                break
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
                logger.warning(f"Semantic Scholar error: {e}")
                break

        status["latency_ms"] = int((time.time() - start_time) * 1000)
        return papers, status


# ==============================================================================
# 2. NORMALIZATION & DEDUPLICATION WITH METADATA MERGING
# ==============================================================================

def normalize_doi(doi_str: Optional[str]) -> Optional[str]:
    """Standardizes DOIs into clean lowercase prefix '10.xxxx/yyyy'."""
    if not doi_str:
        return None
    cleaned = doi_str.strip().lower()
    cleaned = re.sub(r"^https?://(dx\.)?doi\.org/", "", cleaned)
    cleaned = re.sub(r"^doi:\s*", "", cleaned)
    if cleaned.startswith("10.") and "/" in cleaned:
        return cleaned
    return None


def normalize_title(title_str: str) -> str:
    """Normalizes title into lowercase alphanumeric character string for collision matching."""
    if not title_str:
        return ""
    cleaned = title_str.lower()
    cleaned = re.sub(r"[^a-z0-9]", "", cleaned)
    return cleaned


def merge_paper_records(primary: PaperMetadata, incoming: PaperMetadata) -> PaperMetadata:
    """
    Merges complementary metadata from incoming duplicate record into primary record.
    Preserves source provenance and fills gaps without discarding useful fields.
    """
    # 1. Fill missing abstract
    if not primary.abstract and incoming.abstract:
        primary.abstract = incoming.abstract
    elif primary.abstract and incoming.abstract and len(incoming.abstract) > len(primary.abstract):
        # Choose longer, more detailed abstract
        primary.abstract = incoming.abstract

    # 2. Fill missing publication year
    if primary.year is None and incoming.year is not None:
        primary.year = incoming.year

    # 3. Fill missing DOI
    if not primary.doi and incoming.doi:
        primary.doi = normalize_doi(incoming.doi)

    # 4. Fill or improve URL
    if not primary.url and incoming.url:
        primary.url = incoming.url

    # 5. Merge authors list preserving unique author names
    existing_authors_lower = {a.lower() for a in primary.authors}
    for author in incoming.authors:
        if author.lower() not in existing_authors_lower:
            primary.authors.append(author)
            existing_authors_lower.add(author.lower())

    # 6. Keep highest reported citation count
    primary.citation_count = max(primary.citation_count, incoming.citation_count)

    # 7. Merge source provenance
    if incoming.source not in primary.source:
        primary.source = f"{primary.source}+{incoming.source}"

    return primary


def deduplicate_papers(papers: List[PaperMetadata]) -> List[PaperMetadata]:
    """
    Deduplicates paper records using normalized DOI as primary key,
    falling back to normalized title + publication year.
    Merges complementary metadata on duplicate detection.
    """
    unique_by_doi: Dict[str, PaperMetadata] = {}
    unique_by_title_year: Dict[str, PaperMetadata] = {}
    ordered_papers: List[PaperMetadata] = []

    for p in papers:
        # Standardize DOI
        p.doi = normalize_doi(p.doi)

        doi_key = p.doi
        norm_title = normalize_title(p.title)
        title_year_key = f"{norm_title}_{p.year}" if p.year else norm_title

        existing_match: Optional[PaperMetadata] = None

        if doi_key and doi_key in unique_by_doi:
            existing_match = unique_by_doi[doi_key]
        elif title_year_key in unique_by_title_year:
            existing_match = unique_by_title_year[title_year_key]

        if existing_match:
            # Merge complementary metadata
            merge_paper_records(existing_match, p)
            if doi_key and not existing_match.doi:
                unique_by_doi[doi_key] = existing_match
        else:
            if doi_key:
                unique_by_doi[doi_key] = p
            unique_by_title_year[title_year_key] = p
            ordered_papers.append(p)

    return ordered_papers


# ==============================================================================
# 3. DETERMINISTIC RELEVANCE FILTERING & RANKING
# ==============================================================================

def extract_facet_keywords(context_text: str) -> Set[str]:
    """Extracts informative domain keywords and known AI/ML facets."""
    tokens = set(re.findall(r"\b[a-zA-Z0-9-]{3,}\b", context_text.lower()))
    content_tokens = {t for t in tokens if t not in QUERY_STOPWORDS}
    return content_tokens


def is_obviously_off_topic(title: str, abstract: str, target_keywords: Set[str]) -> Tuple[bool, str]:
    """
    Detects if a paper belongs to an obviously unrelated discipline
    (e.g., historical philosophy like 'Theory of Moral Sentiments' or clinical surgery)
    unless the research problem explicitly asks for it.
    """
    full_text = f"{title.lower()} {abstract.lower()}"

    for category, trigger_phrases in OFF_TOPIC_DOMAINS.items():
        for phrase in trigger_phrases:
            if phrase in full_text:
                # If target research problem didn't ask for this domain, filter it
                if not any(phrase_word in target_keywords for phrase_word in phrase.split()):
                    return True, f"Off-topic domain detected ({category}: '{phrase}')"

    return False, ""


def compute_deterministic_relevance(
    paper: PaperMetadata,
    research_question: str,
    task_title: str,
    task_description: str,
    current_year: int = 2026
) -> Tuple[float, str, str]:
    """
    Computes a transparent, deterministic relevance score (0.0 to 1.0) and rationale.
    Examines title matches, abstract matches, model/benchmark mentions, and recency.
    Categorizes confidence into high, medium, borderline, or filtered.
    """
    combined_context = f"{research_question} {task_title} {task_description}"
    target_keywords = extract_facet_keywords(combined_context)

    title_lower = paper.title.lower()
    abstract_lower = paper.abstract.lower()

    # 1. Check for obviously off-topic papers
    off_topic, reason = is_obviously_off_topic(paper.title, paper.abstract, target_keywords)
    if off_topic:
        return 0.05, f"Filtered: {reason}", "filtered"

    # 2. Topic keyword matches
    title_matches = [kw for kw in target_keywords if kw in title_lower]
    abstract_matches = [kw for kw in target_keywords if kw in abstract_lower]

    # 3. Model and Dataset facet matches
    model_matches = [m for m in KNOWN_MODELS if m in combined_context.lower() and (m in title_lower or m in abstract_lower)]
    dataset_matches = [d for d in KNOWN_DATASETS if d in combined_context.lower() and (d in title_lower or d in abstract_lower)]
    metric_matches = [m for m in KNOWN_METRICS if m in combined_context.lower() and (m in title_lower or m in abstract_lower)]

    # Scoring weights:
    # Title match: 3.0x, Abstract match: 1.0x, Model match: 2.0x, Dataset match: 1.5x
    score_points = (len(title_matches) * 3.0) + (len(abstract_matches) * 1.0) + (len(model_matches) * 2.0) + (len(dataset_matches) * 1.5)
    max_expected = max(5.0, len(target_keywords) * 1.8)
    text_match = min(1.0, score_points / max_expected)

    # Recency factor (gradual decay over 10 years, minimum 0.5)
    paper_year = paper.year or 2020
    age = max(0, current_year - paper_year)
    recency_factor = max(0.5, 1.0 - (age * 0.05))

    # Citation signal (soft log scale, max 0.1 bonus)
    citation_bonus = min(0.1, (paper.citation_count / 1000.0) * 0.1) if paper.citation_count else 0.0

    final_score = round(min(1.0, (text_match * 0.85 * recency_factor) + citation_bonus), 3)

    # Explanation building
    matched_elements = []
    if title_matches:
        matched_elements.append(f"title matched {title_matches[:3]}")
    if model_matches:
        matched_elements.append(f"models {model_matches[:2]}")
    if dataset_matches:
        matched_elements.append(f"datasets {dataset_matches[:2]}")
    if abstract_matches:
        matched_elements.append(f"abstract keywords {len(abstract_matches)}")

    explanation = "; ".join(matched_elements) if matched_elements else "weak semantic alignment"

    # Confidence classification
    if final_score >= 0.60:
        confidence = "high"
    elif final_score >= 0.35:
        confidence = "medium"
    elif final_score >= settings.LITERATURE_MIN_RELEVANCE_THRESHOLD:
        confidence = "borderline"
    else:
        confidence = "filtered"

    return final_score, explanation, confidence


def filter_and_rank_papers(
    papers: List[PaperMetadata],
    research_question: str,
    task_title: str,
    task_description: str,
    min_threshold: Optional[float] = None,
    limit: int = 10
) -> List[PaperMetadata]:
    """
    Ranks papers descending by relevance score and filters out clearly off-topic papers.
    Retains borderline papers with lower confidence rather than silently discarding them.
    """
    threshold = min_threshold if min_threshold is not None else settings.LITERATURE_MIN_RELEVANCE_THRESHOLD

    scored_papers: List[PaperMetadata] = []
    for p in papers:
        score, explanation, conf = compute_deterministic_relevance(
            paper=p,
            research_question=research_question,
            task_title=task_title,
            task_description=task_description
        )
        p.relevance_score = score
        p.relevance_explanation = explanation
        p.confidence = conf

        # Filter out definitely off-topic papers
        if score >= threshold:
            scored_papers.append(p)
        else:
            logger.debug(f"Filtered out off-topic paper: '{p.title[:50]}' (Score: {score}, Reason: {explanation})")

    # Sort descending by relevance score, secondary by citation count
    scored_papers.sort(key=lambda x: (x.relevance_score or 0.0, x.citation_count), reverse=True)
    return scored_papers[:limit]


# ==============================================================================
# 4. STRUCTURED RESEARCH EVIDENCE EXTRACTION & VALIDATION
# ==============================================================================

CANONICAL_METRIC_MAP = {
    "accuracy": "accuracy",
    "accuracies": "accuracy",
    "acc": "accuracy",
    "precision": "precision",
    "precisions": "precision",
    "recall": "recall",
    "recalls": "recall",
    "f1": "f1_score",
    "f1-score": "f1_score",
    "f1_score": "f1_score",
    "f1 score": "f1_score",
    "macro-f1": "f1_score",
    "micro-f1": "f1_score",
    "weighted-f1": "f1_score",
    "auc": "auc",
    "roc-auc": "auc",
    "roc_auc": "auc",
    "roc auc": "auc",
    "pr-auc": "auc",
    "bleu": "bleu",
    "bleu-1": "bleu",
    "bleu-2": "bleu",
    "bleu-3": "bleu",
    "bleu-4": "bleu",
    "bleu score": "bleu",
    "rouge": "rouge",
    "rouge-1": "rouge",
    "rouge-2": "rouge",
    "rouge-l": "rouge",
    "rouge_l": "rouge",
    "rouge-w": "rouge",
    "rouge score": "rouge",
    "rmse": "rmse",
    "root mean squared error": "rmse",
    "mse": "mse",
    "mean squared error": "mse",
    "mae": "mae",
    "mean absolute error": "mae",
}


def extract_metric_results_from_text(
    text: str,
    available_datasets: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extracts quantitative metric outcomes from text and maps each numerical value
    strictly to its corresponding metric without crossing metric-value boundaries.
    Preserves exact relationships from source text. Never invents results.
    Returns: (reported_results, evaluation_metrics)
    """
    if not text:
        return [], []

    reported_results: List[Dict[str, Any]] = []
    detected_metrics: Set[str] = set()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    metric_pattern = r"(accuracy|precision|recall|f1(?:-score|_score|\s+score)?|macro-f1|micro-f1|weighted-f1|roc-auc|pr-auc|auc|bleu(?:-[0-9])?|rouge(?:-[0-9a-zA-Z]+)?|rmse|mse|mae)"
    val_pattern = r"(\d{1,3}(?:\.\d+)?%|\b0\.\d{1,4}\b|\b\d{1,3}\.\d{1,3}\b)"

    p1 = re.compile(
        rf"{val_pattern}\s*(?:[-–—,\s]|in\s+|of\s+|on\s+)*{metric_pattern}\b",
        re.IGNORECASE
    )
    p2 = re.compile(
        rf"\b{metric_pattern}\s*(?:score|rate)?\s*(?:of|:|is|reached|achieved|=|was)?\s*{val_pattern}",
        re.IGNORECASE
    )

    for sentence in sentences:
        s_clean = sentence.strip()
        if not s_clean:
            continue

        s_lower = s_clean.lower()
        for m_key, canonical in CANONICAL_METRIC_MAP.items():
            if re.search(r"\b" + re.escape(m_key) + r"\b", s_lower):
                detected_metrics.add(canonical)

        matched_spans: List[Tuple[int, int, str, str]] = []
        val_spans_used: Set[Tuple[int, int]] = set()

        # Pattern 1: Value followed by Metric (e.g., "90.4% accuracy", "88% recall", "87.3% precision")
        for m in p1.finditer(s_clean):
            val_str = m.group(1).strip()
            raw_metric = m.group(2).strip().lower()
            canonical_metric = CANONICAL_METRIC_MAP.get(raw_metric, raw_metric)
            v_start = m.start(1)
            v_end = m.end(1)
            matched_spans.append((v_start, v_end, canonical_metric, val_str))
            val_spans_used.add((v_start, v_end))

        # Pattern 2: Metric followed by Value (e.g., "an F1-score of 87.7%", "accuracy of 92%")
        for m in p2.finditer(s_clean):
            raw_metric = m.group(1).strip().lower()
            val_str = m.group(2).strip()
            canonical_metric = CANONICAL_METRIC_MAP.get(raw_metric, raw_metric)
            v_start = m.start(2)
            v_end = m.end(2)
            if not any(abs(v_start - used_s) <= 1 for used_s, used_e in val_spans_used):
                matched_spans.append((v_start, v_end, canonical_metric, val_str))
                val_spans_used.add((v_start, v_end))

        matched_spans.sort(key=lambda x: x[0])

        associated_dataset = None
        if available_datasets:
            for d in available_datasets:
                if d.lower() in s_lower:
                    associated_dataset = d
                    break

        for _, _, metric, val in matched_spans:
            res_item: Dict[str, Any] = {
                "metric": metric,
                "value": val,
                "source_text": s_clean
            }
            if associated_dataset:
                res_item["dataset"] = associated_dataset
            reported_results.append(res_item)

    return reported_results, sorted(list(detected_metrics))


def extract_datasets_from_text(text: str) -> List[str]:
    """
    Extracts benchmark and empirical dataset names ONLY when explicitly supported
    by the abstract or paper text. Returns an empty list when no dataset can be reliably identified.
    Never invents dataset names (e.g. does NOT assume Twitter unless explicitly 'Twitter dataset').
    """
    if not text:
        return []

    found_datasets: List[str] = []
    seen_lower: Set[str] = set()

    excluded_words = {
        "the", "our", "this", "we", "a", "an", "in", "on", "for", "with", "by", "from", "to",
        "table", "figure", "section", "model", "models", "baseline", "baselines", "results",
        "result", "state", "sota", "accuracy", "f1", "precision", "recall", "loss", "none",
        "transformer", "bert", "roberta", "svm", "lstm", "cnn", "gpt", "llm", "ai", "ml", "nlp",
        "gpu", "cpu", "api", "url", "doi", "english", "arabic", "chinese", "french", "german",
        "spanish", "method", "methods", "approach", "paper", "study", "analysis", "experiments",
        "experiment", "evaluation", "benchmark", "dataset", "corpus", "data", "classification",
        "sentiment", "performance", "task", "tasks", "deep", "learning", "all", "both", "each",
        "few", "many", "several", "various", "multiple", "existing", "standard", "popular"
    }

    # 1. Pattern A: Explicit dataset suffix (e.g., "ASTD dataset", "SST-2 benchmark", "TEAD corpus")
    suffix_pattern = re.compile(
        r"\b([A-Z0-9][A-Za-z0-9_.-]{1,20})\s+(?:dataset|datasets|benchmark|benchmarks|corpus|corpora)\b",
        re.IGNORECASE
    )
    for m in suffix_pattern.finditer(text):
        candidate = m.group(1).strip()
        if candidate.lower() not in excluded_words and len(candidate) >= 2:
            if candidate.lower() not in seen_lower:
                found_datasets.append(candidate)
                seen_lower.add(candidate.lower())

    # 2. Pattern B: Explicit evaluation preposition followed by acronyms/names
    # e.g., "evaluated on ASTD, ASAD, and TEAD", "benchmarked on SST-2 and IMDb"
    prep_pattern = re.compile(
        r"\b(?:evaluated\s+on|evaluated\s+using|tested\s+on|benchmarked\s+on|trained\s+on)\s+([^.;\n]+)",
        re.IGNORECASE
    )
    for m in prep_pattern.finditer(text):
        raw_list_str = m.group(1).strip()
        items = re.split(r",\s*and\s+|,?\s+and\s+|,?\s+or\s+|,", raw_list_str)
        for item in items:
            cand = item.strip(" .,;:()[]")
            if cand.lower().startswith("the "):
                cand = cand[4:].strip()
            # If followed by 'datasets' or 'benchmarks', trim it
            cand = re.sub(r"\s+(?:datasets?|benchmarks?|corpus|corpora)$", "", cand, flags=re.IGNORECASE).strip()
            if cand and cand.lower() not in excluded_words and len(cand) >= 2:
                if cand.isupper() or any(c.isupper() for c in cand) or "-" in cand:
                    if cand.lower() not in seen_lower:
                        found_datasets.append(cand)
                        seen_lower.add(cand.lower())

    # 3. Pattern C: Known canonical benchmark datasets IF present with exact word boundaries
    BENCHMARK_DATASETS = [
        ("imdb", "IMDb"),
        ("sst-2", "SST-2"),
        ("sst-5", "SST-5"),
        ("sst", "SST"),
        ("glue", "GLUE"),
        ("superglue", "SuperGLUE"),
        ("squad", "SQuAD"),
        ("mnli", "MNLI"),
        ("qnli", "QNLI"),
        ("mrpc", "MRPC"),
        ("qqp", "QQP"),
        ("ag news", "AG News"),
        ("financial phrasebank", "Financial PhraseBank"),
        ("semeval", "SemEval"),
        ("multinli", "MultiNLI")
    ]
    for key, display_name in BENCHMARK_DATASETS:
        if re.search(r"\b" + re.escape(key) + r"\b", text, re.IGNORECASE):
            if display_name.lower() not in seen_lower:
                found_datasets.append(display_name)
                seen_lower.add(display_name.lower())

    # 4. Strict verification: every returned dataset must appear in source text
    verified_datasets = [
        d for d in found_datasets
        if re.search(r"\b" + re.escape(d) + r"\b", text, re.IGNORECASE)
    ]

    return verified_datasets


def extract_models_from_text(text: str) -> List[str]:
    """Extracts known machine learning models or algorithms mentioned in text."""
    if not text:
        return []
    text_lower = text.lower()
    models = []
    seen = set()
    for m in KNOWN_MODELS:
        if re.search(r"\b" + re.escape(m) + r"\b", text_lower):
            display = m.upper() if len(m) <= 4 and m not in ("bert", "bart", "t5") else m.title()
            if display == "Bert":
                display = "BERT"
            elif display == "Svm":
                display = "SVM"
            elif display == "Cnn":
                display = "CNN"
            elif display == "Lstm":
                display = "LSTM"
            elif display == "Gpt":
                display = "GPT"
            if display.lower() not in seen:
                models.append(display)
                seen.add(display.lower())
    return models


def deduplicate_evidence(evidence_list: List[ExtractedEvidence]) -> List[ExtractedEvidence]:
    """
    Deduplicates evidence records using normalized DOI as primary key,
    falling back to normalized title + publication year.
    Merges complementary findings, metrics, datasets, and reported results.
    """
    unique_by_doi: Dict[str, ExtractedEvidence] = {}
    unique_by_title_year: Dict[str, ExtractedEvidence] = {}
    ordered_evidence: List[ExtractedEvidence] = []

    for ev in evidence_list:
        doi_val = normalize_doi(ev.doi) or normalize_doi(ev.paper_id)
        if doi_val:
            ev.doi = doi_val

        doi_key = doi_val
        norm_title = normalize_title(ev.title or "")
        title_year_key = f"{norm_title}_{ev.year}" if (norm_title and ev.year) else norm_title

        existing: Optional[ExtractedEvidence] = None
        if doi_key and doi_key in unique_by_doi:
            existing = unique_by_doi[doi_key]
        elif title_year_key and title_year_key in unique_by_title_year:
            existing = unique_by_title_year[title_year_key]

        if existing:
            # Merge complementary data
            existing_results_keys = {(r.get("metric"), r.get("value")) for r in existing.reported_results}
            for r in ev.reported_results:
                if (r.get("metric"), r.get("value")) not in existing_results_keys:
                    existing.reported_results.append(r)
                    existing_results_keys.add((r.get("metric"), r.get("value")))

            existing_models_lower = {m.lower() for m in existing.models_or_algorithms}
            for m in ev.models_or_algorithms:
                if m.lower() not in existing_models_lower:
                    existing.models_or_algorithms.append(m)
                    existing_models_lower.add(m.lower())

            existing_datasets_lower = {d.lower() for d in existing.datasets}
            for d in ev.datasets:
                if d.lower() not in existing_datasets_lower:
                    existing.datasets.append(d)
                    existing_datasets_lower.add(d.lower())

            existing_metrics_lower = {m.lower() for m in existing.evaluation_metrics}
            for m in ev.evaluation_metrics:
                if m.lower() not in existing_metrics_lower:
                    existing.evaluation_metrics.append(m)
                    existing_metrics_lower.add(m.lower())

            existing_findings = set(existing.key_findings)
            for f in ev.key_findings:
                if f not in existing_findings:
                    existing.key_findings.append(f)
                    existing_findings.add(f)

            if not existing.research_objective and ev.research_objective:
                existing.research_objective = ev.research_objective
            elif len(ev.research_objective) > len(existing.research_objective):
                existing.research_objective = ev.research_objective

            if ev.evidence_status == "extracted" and existing.evidence_status != "extracted":
                existing.evidence_status = "extracted"
                existing.evidence_source = ev.evidence_source

            if doi_key and not existing.doi:
                existing.doi = doi_key
                unique_by_doi[doi_key] = existing
        else:
            if doi_key:
                unique_by_doi[doi_key] = ev
            if title_year_key:
                unique_by_title_year[title_year_key] = ev
            ordered_evidence.append(ev)

    return ordered_evidence


def validate_extracted_evidence(evidence: ExtractedEvidence, source_text: str) -> ExtractedEvidence:
    """
    Validates extracted evidence against the source text:
    - Every numerical value must literally appear in the source text.
    - Metric names must match the source text.
    - Dataset names must be supported by the source text.
    - Missing information remains empty rather than fabricated.
    - Evidence status accurately reflects abstract-only ('partial') vs metadata-only ('unavailable').
    Logs validation rejections without exposing API keys.
    """
    source_lower = source_text.lower()

    # 1. Validate reported numerical results
    validated_results: List[Dict[str, Any]] = []
    for r in evidence.reported_results:
        val = str(r.get("value", "")).strip()
        metric = str(r.get("metric", "")).strip()

        if not val or val not in source_text:
            logger.warning(
                f"Validation rejection for paper '{evidence.paper_id}': "
                f"Numerical value '{val}' not found in source text."
            )
            continue

        metric_variants = [metric, metric.replace("_", "-"), metric.replace("_", " ")]
        if not any(v in source_lower for v in metric_variants):
            logger.warning(
                f"Validation rejection for paper '{evidence.paper_id}': "
                f"Metric '{metric}' not supported by source text context."
            )
            continue

        validated_results.append(r)

    evidence.reported_results = validated_results

    # 2. Validate datasets
    validated_datasets: List[str] = []
    for d in evidence.datasets:
        if d.lower() in source_lower and d not in ("Reported Benchmark", "Unknown", "None"):
            validated_datasets.append(d)
        else:
            logger.warning(
                f"Validation rejection for paper '{evidence.paper_id}': "
                f"Dataset '{d}' not supported in source text."
            )
    evidence.datasets = validated_datasets

    # 3. Validate models
    evidence.models_or_algorithms = [
        m for m in evidence.models_or_algorithms if m.lower() in source_lower
    ]

    # 4. Validate evaluation metrics
    evidence.evaluation_metrics = [
        m for m in evidence.evaluation_metrics
        if any(v in source_lower for v in [m.lower(), m.lower().replace("_", "-"), m.lower().replace("_", " ")])
    ]

    # 5. Validate evidence source and status
    if not source_text or len(source_text.strip()) < 30:
        evidence.evidence_status = "unavailable"
        evidence.evidence_source = "metadata"
    else:
        if evidence.evidence_source != "full_text":
            evidence.evidence_source = "abstract"
            evidence.evidence_status = "partial"

    return evidence


def extract_evidence_from_paper(paper: PaperMetadata, task_context: str = "") -> ExtractedEvidence:
    """
    Extracts structured, verifiable empirical research evidence from paper metadata and abstract text.
    Associates numerical values with their exact metrics without fabrication.
    Extracts datasets only when supported by source text.
    Marks abstract-only evidence as 'partial' and 'abstract'.
    """
    full_text = f"{paper.title}. {paper.abstract}".strip()

    models_found = extract_models_from_text(full_text)
    datasets_found = extract_datasets_from_text(full_text)
    reported_results, metrics_found = extract_metric_results_from_text(
        text=full_text,
        available_datasets=datasets_found
    )

    objective = ""
    if paper.abstract and len(paper.abstract.strip()) > 10:
        sentences = re.split(r"(?<=[.!?])\s+", paper.abstract.strip())
        objective = sentences[0].strip() if sentences else paper.title
    else:
        objective = f"Investigates {paper.title}."

    findings: List[str] = []
    if paper.abstract:
        for s in re.split(r"(?<=[.!?])\s+", paper.abstract):
            s_clean = s.strip()
            if any(term in s_clean.lower() for term in ["we find", "results show", "outperforms", "improves", "demonstrates", "conclude", "propose", "achieves", "achieved"]):
                findings.append(s_clean)
        if not findings and len(paper.abstract) > 30:
            findings.append(paper.abstract[:180] + "...")

    limitations: List[str] = []
    if not paper.abstract:
        limitations.append("Full paper text and abstract unavailable from source repository; metadata-only record.")
    else:
        limitations.append("Evidence derived from published paper abstract; full PDF text analysis not executed.")

    if not paper.abstract or len(paper.abstract.strip()) < 30:
        evidence_status = "unavailable"
        evidence_source = "metadata"
    else:
        evidence_status = "partial"
        evidence_source = "abstract"

    evidence = ExtractedEvidence(
        paper_id=paper.paper_id,
        title=paper.title,
        doi=paper.doi,
        year=paper.year,
        research_objective=objective,
        models_or_algorithms=models_found,
        datasets=datasets_found,
        evaluation_metrics=metrics_found,
        reported_results=reported_results,
        key_findings=findings[:3],
        limitations=limitations,
        evidence_source=evidence_source,
        evidence_status=evidence_status
    )

    return validate_extracted_evidence(evidence, source_text=full_text)


# ==============================================================================
# 5. TASK SUCCESS CRITERIA VALIDATION
# ==============================================================================

def validate_task_success_criteria(
    task_spec: Dict[str, Any],
    papers: List[PaperMetadata],
    evidence: List[ExtractedEvidence]
) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Validates whether retrieved papers and evidence satisfy the subtask success criteria.
    Parses target count, metadata completeness, and evidence coverage.
    Returns: (status: completed|partial|failed, success_criteria_results, warnings)
    """
    criteria_text = task_spec.get("success_criteria", "")
    warnings: List[str] = []

    # 1. Parse target paper count (e.g., "At least 5 relevant papers...", "15 papers...")
    target_count = 5
    count_match = re.search(r"(\d+)\s*(?:relevant\s*)?papers?", criteria_text, re.IGNORECASE)
    if count_match:
        target_count = int(count_match.group(1))

    actual_count = len(papers)

    # 2. Check metadata coverage (abstracts and authors)
    papers_with_abstracts = sum(1 for p in papers if bool(p.abstract and len(p.abstract) > 30))
    papers_with_authors = sum(1 for p in papers if bool(p.authors))
    abstract_coverage = round(papers_with_abstracts / actual_count, 2) if actual_count > 0 else 0.0

    # 3. Check evidence extraction
    evidence_extracted_count = sum(1 for e in evidence if e.evidence_status in ("extracted", "partial"))

    criteria_passed = (actual_count >= target_count) and (abstract_coverage >= 0.5)

    if actual_count == 0:
        status = "failed"
        warnings.append(f"Zero relevant papers retrieved (Target was {target_count}).")
        details = f"Failed: 0 relevant papers found against target of {target_count}."
    elif actual_count < target_count:
        status = "partial"
        warnings.append(f"Retrieved {actual_count} relevant papers, which is below the requested target of {target_count}.")
        details = f"Partial: Found {actual_count} relevant papers (Target was {target_count}). Abstract coverage: {int(abstract_coverage * 100)}%."
    else:
        status = "completed"
        details = f"Completed: Satisfied target count ({actual_count}/{target_count}) with {int(abstract_coverage * 100)}% abstract coverage."

    if abstract_coverage < 0.5 and actual_count > 0:
        warnings.append(f"Low abstract coverage: Only {papers_with_abstracts}/{actual_count} papers include full abstracts.")

    results_summary = {
        "criteria_text": criteria_text,
        "target_count": target_count,
        "actual_count": actual_count,
        "abstract_coverage": abstract_coverage,
        "evidence_count": evidence_extracted_count,
        "passed": criteria_passed,
        "details": details
    }

    return status, results_summary, warnings
