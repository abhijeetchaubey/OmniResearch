# OmniResearch

OmniResearch is an AI-powered research orchestration system that turns a single research question into a structured, multi-step workflow for literature review, dataset discovery, experimentation, validation, and final reporting.

## The problem we are solving

Research work is often fragmented across multiple disconnected activities:

- Searching for relevant papers and sources takes time and manual effort.
- Selecting the right dataset and evaluation strategy is often inconsistent.
- Experiment execution and result tracking can become difficult to manage.
- Dependencies between tasks are easy to miss without a formal workflow.
- Producing a polished research summary requires coordination across many stages.

OmniResearch solves this by giving researchers a guided, agent-based system that decomposes a research goal into actionable tasks and orchestrates them in a structured workflow.

## What we are building

We are building an intelligent research assistant that can:

- accept a high-level research question,
- generate a structured research plan,
- assign tasks to specialized agents,
- manage task dependencies and sequencing,
- coordinate literature review, dataset selection, analysis, and validation,
- generate a final research output that is ready for further review.

The system is designed to support research workflows that are scalable, traceable, and easier to extend over time.

## Tech stack

OmniResearch is built with a modern Python AI stack:

- Python as the core application language
- FastAPI for the backend API
- Uvicorn as the ASGI server
- Pydantic and Pydantic Settings for data validation and config management
- LangChain for LLM integration and prompt orchestration
- LangGraph for workflow orchestration and task routing
- Groq as the LLM provider
- dotenv for environment-based configuration
- pytest and unittest for testing

## Architecture overview

The system follows a workflow-first design:

1. A user submits a research question through the API.
2. The Planner Agent creates a structured research plan with subtasks and dependencies.
3. The workflow router evaluates which task is ready to run next.
4. Each stage is assigned to a specialized agent responsible for a specific research activity.
5. The final state is maintained in a graph-based workflow that tracks progress and generated artifacts.

## Project structure

```text
OmniResearch/
├── backend/
│   ├── agents/
│   ├── api/
│   ├── config/
│   ├── graph/
│   └── main.py
├── tests/
├── .env.example
├── requirements.txt
├── README.md
├── .gitignore
└── OmniResearch_Developer_Technical_Specification.docx
```

## How the system works

The key idea is to transform an unstructured research prompt into a clear, executable research plan.

- The Planner Agent decomposes the question into manageable subtasks.
- Each plan includes three main objectives.
- Each subtask has a title, description, expected output, success criteria, and dependencies.
- The routing layer ensures tasks are only executed when their prerequisites are satisfied.
- Specialized agents handle distinct phases of the research process.
- The workflow can be extended and adapted as more research capabilities are added.

Literature results include a source-linked comparison table with publication year, authors, methods, reported findings, results, datasets, and limitations. Numerical results retain their supporting source text. Dataset names and their availability are not inferred: availability is marked as unverified unless separately checked. Dataset discovery, experiment execution, verification, and report-writing agents are currently placeholders.

## Getting started

### 1. Set up a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file using `.env.example` and provide your API key and app settings.

```env
GROQ_API_KEY=your_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
API_PORT=8000
```

### 4. Run the backend

```bash
uvicorn backend.main:app --reload
```

The service will be available at:

- http://localhost:8000
- http://localhost:8000/docs
- http://localhost:8000/health

## API

The main research endpoint is:

```http
POST /research/plan
```

This endpoint accepts a research question and returns a structured research plan that can be used by the workflow engine.

## Why this project matters

OmniResearch is designed to reduce the time and complexity involved in turning research questions into action. Instead of managing research as a single manual process, it models the work as an orchestrated system of tasks, dependencies, and specialized AI agents. That makes the workflow more structured, more scalable, and easier to iterate as research needs evolve.

## Roadmap

Planned improvements include:

- richer agent capabilities for literature and dataset retrieval,
- stronger experiment orchestration and evaluation flows,
- more advanced verification and citation support,
- report generation with structured outputs,
- improved observability and workflow analytics.

## License

This project is currently in active development for research and experimentation purposes.
