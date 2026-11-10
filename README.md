# Risk News Monitor

[![CI](https://github.com/pranav2258/risk-news-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/pranav2258/risk-news-monitor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1c3c3c)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A LangGraph agent that reads a news feed, matches articles to monitored counterparties,
assesses credit / legal / operational / market risk, and **pauses for a human analyst**
before any high-risk or suspicious alert is published.

It runs fully offline with no API key: the model step is a deterministic, rule-based
stand-in behind the same interface a real LLM would use, so tests and evaluations give
the same answer every run.

![Architecture](docs/architecture.png)

![A run of the demo: the agent pauses on an article containing a prompt-injection attempt](docs/demo-run.png)

> All companies, people and articles in `data/` are synthetic.

## Quick start

Needs Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                             # 21 tests
python evals/run_evals.py          # quality metrics + pass/fail gates
python run_demo.py --reset         # you approve/reject alerts in the terminal
python run_demo.py --reset --auto-approve   # same, no prompts
```

## How the agent works

```mermaid
flowchart TD
    S([START]) --> L[load_and_screen<br/>load article, strip prompt injection]
    L --> M[match_counterparty]
    M -- not_relevant --> K[skip] --> E([END])
    M -- relevant --> R[retrieve_context<br/>profile + prior alerts + related chunks<br/>assembled within a token budget]
    R --> A[assess_risk<br/>structured output]
    A --> V[validate_output<br/>schema + grounding checks]
    V -- auto_publish --> P[publish]
    V -- needs_review --> H[human_review<br/>interrupt: graph pauses]
    H --> P
    P --> E
```

- **State** (`RiskState` in `monitor/nodes.py`) is a typed dict passed between nodes. Each node
  returns only the keys it changes. `audit_log` uses a reducer (`operator.add`), so every step
  appends to it and the run leaves a full audit trail.
- **Routing**: two conditional edges. Articles with no monitored counterparty are skipped.
  An article goes to human review if severity is high, if prompt injection was detected, or if
  the output failed validation.
- **Human-in-the-loop**: `human_review` calls `interrupt()`. The checkpointer saves the state and
  the run stops. The caller resumes it with `Command(resume={"approved": True, ...})`.
- **Checkpointing**: one `thread_id` per article. The demo uses `InMemorySaver`; swapping in a
  SQLite or Postgres checkpointer would let a review happen later, in another process.

## Project layout

| Path | What it does |
|---|---|
| `monitor/data.py` | SQLite store: validation, de-duplication, incremental loads, memory queries |
| `monitor/retrieval.py` | Sentence chunking with overlap, TF-IDF vectors, filtered similarity search |
| `monitor/llm.py` | Prompt, context builder (token budget), output schema, offline assessor |
| `monitor/guardrails.py` | Prompt-injection screening, schema and grounding validation |
| `monitor/nodes.py` | Agent state, node functions, routing functions |
| `monitor/graph.py` | LangGraph wiring, human-review interrupt, checkpointer |
| `monitor/runner.py` | Runs articles through the graph, handles pause/resume and failures |
| `monitor/tracing.py` | Per-node trace events (latency, status, tokens) to JSONL |
| `evals/` | Labelled eval set and scoring script with quality gates |
| `tests/` | Unit tests and end-to-end graph tests |
| `.github/workflows/ci.yml` | Runs tests, eval gates and a demo smoke test on every push |

## What it demonstrates

| Area | Where |
|---|---|
| Data engineering: validation, dedupe by URL and by title, idempotent upserts, incremental processing, failure recovery | `data.py`, `runner.py` |
| SQL: schema design, `UNIQUE`/upserts, a join with a date window for memory | `data.py` |
| RAG: chunking, metadata enrichment, vector search with metadata filters, no look-ahead | `retrieval.py` |
| Context engineering: priority-ordered context within a token budget, records what was dropped | `llm.py` `build_context` |
| Prompt engineering: versioned prompt, grounding rules, structured JSON output, untrusted-data framing | `llm.py` |
| Memory: short-term = graph state per thread; long-term = published alerts per counterparty | `nodes.py`, `data.py` |
| Guardrails: injection screening, schema validation, evidence must be quoted from the article | `guardrails.py` |
| Agent orchestration: typed state, reducer, conditional routing, interrupt/resume, checkpointer | `graph.py` |
| Evaluation: labelled set, metrics, CI-style gates, failure list, version comparison | `evals/` |
| Observability: per-node latency, outcome, token estimate; audit trail in state | `tracing.py` |
| Testing: 21 pytest tests incl. pause/resume and regression tests for eval-found bugs | `tests/` |
| CI: tests + eval quality gates + demo smoke test on Python 3.10 and 3.12 | `.github/workflows/ci.yml` |

## Evaluation results

`python evals/run_evals.py --version v1|v2|v3` reproduces each row.

| Version | Change | Assessment accuracy | High-severity precision | Routing accuracy | Gates |
|---|---|---|---|---|---|
| v1 | baseline keyword rules | 0.80 | 0.60 | 0.91 | fail |
| v2 | ignore negated signals ("does not expect any default") | 0.90 | 0.75 | 1.00 | fail |
| v3 | escalate on memory only for the *same developing story* | 1.00 | 1.00 | 1.00 | pass |

High-severity recall, grounding pass rate and injection detection were 1.00 in every version.

What the evals found:

1. **v1, negation.** "The company does not expect any default" was scored as a high credit event
   and wrongly sent for review. Fix: ignore a signal if a negation word appears just before it.
2. **v2, over-escalation.** A terminal fire was escalated to high because of an unrelated port-strike
   alert a month earlier: the memory rule only checked "same risk type". Retrieval scores separated
   the cases clearly (same story 0.46, unrelated 0.15 to 0.26), so v3 escalates only when the earlier
   alert's article is retrieved with a score of at least 0.35.

Both fixes have regression tests in `tests/test_components.py`.

**Caveat:** 11 labelled cases is a smoke test, not a benchmark. The 0.35 threshold was picked from one
positive example and would need a much larger, analyst-labelled set before relying on it.

## Limitations and next steps

- **Real LLM.** Implement `Assessor.assess()` with a model client: send `prompt`, parse the JSON into
  `Assessment`. The validation node already rejects ungrounded or malformed output. Re-run the evals
  and compare against the rule-based baseline.
- **Real embeddings.** Replace TF-IDF with an embedding model and a vector store (pgvector, Chroma,
  Snowflake/Databricks vector search); keep the same metadata filters.
- **Persistent checkpointer** (SQLite/Postgres) so reviews survive restarts.
- **Multiple counterparties per article**: fan out one assessment per counterparty.
- **Observability platform**: send traces to Galileo or LangSmith instead of a JSONL file, and track
  quality metrics on live traffic, not just the offline eval set.
- **Access control**: filter what each user's agent can retrieve by entitlements.
- **Bigger eval set** labelled by risk analysts, with LLM-as-judge for rationale quality.

## Development notes

Developed with help from an AI coding assistant. Every change is covered by the test suite and the
eval gates, which run in CI on each push.

Licensed under the MIT License (see `LICENSE`).
