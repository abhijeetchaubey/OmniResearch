# OmniResearch Feature Review and Work Report

**Date:** November 24, 2026

## Summary

The project already had a structured research-planning workflow and source-grounded literature evidence extraction. The review found that the requested comparison table and explicit three-objective plan were missing, literature summaries were not surfaced in the research API response, and dataset availability was not clearly reported. Those gaps have been addressed with small changes to the existing planner, literature agent, and API.

## What Was Already Working

- Research questions are turned into plans with subtasks, dependencies, expected outputs, and success criteria.
- Scholarly papers can be retrieved from OpenAlex, arXiv, and Semantic Scholar.
- Paper records include publication year, authors, title, source, and available links.
- Literature evidence extraction records methods, datasets, reported results, findings, and limitations.
- Extracted numeric results and dataset names are checked against source text; missing data is not meant to be invented.
- The project has tests for planning, literature retrieval/evidence extraction, routing, and workflow behavior.

## Changes Made

- Added an exactly-three-objectives field to research plans, with instructions for the planner and objectives in the fallback plan.
- Added a comparative literature table with year, author, method, reported finding, results and supporting source text, datasets, limitations, and source link.
- Exposed generated literature summaries and comparison tables through the `literature_reviews` field in the research API response.
- Made the dataset caveat explicit: dataset names not found in an abstract are shown as not identified, and dataset access/availability is marked unverified.
- Excluded the generic phrase “named benchmark” from being mistaken for a dataset.
- Updated the README and added focused regression tests for the requested behavior.

## Remaining Limitations

- Dataset discovery, experiment execution, analysis, verification, and report-writing agents are still placeholders; this change does not implement those integrations.
- Evidence is generally based on abstracts and repository metadata. Full-paper PDF parsing is not implemented.
- Mentioning a dataset in a paper does not confirm that the dataset is accessible; availability remains unverified.
- The note “To learn and” in the supplied image is incomplete and was not interpreted as a requirement.

## Validation

- `python -m pytest -q` — **32 passed**
- `git diff --check` — **passed**

## Security Note

An API key was pasted into the conversation earlier. Revoke/rotate that key and use the replacement in the local, ignored `.env` file. No key is included in this report.
