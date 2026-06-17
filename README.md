# AI Financial Document Analyst

A RAG-powered platform that ingests 10-K filings and earnings call
transcripts, extracts financial metrics, analyses management tone, tracks
risk factor changes across periods, benchmarks competitors, and generates
grounded investment memos.

## Features

- **Document ingestion** (`app/ingestion/`): parses PDF 10-Ks and `.txt`
  earnings call transcripts into structured sections (financial statements,
  MD&A, risk factors, forward guidance, transcript Q&A).
- **RAG pipeline** (`app/rag/`): chunks every section, embeds it locally
  (sentence-transformers), and stores it in a Chroma vector store with
  metadata filters (ticker, period, section type) for retrieval-augmented
  Q&A.
- **Financial metric extraction** (`app/extraction/metrics.py`): revenue,
  margins, cash flow, debt, and forward guidance, plus YoY/QoQ comparisons
  (`app/extraction/comparison.py`).
- **Management tone analysis** (`app/analysis/tone.py`): sentiment,
  confidence score, hedging vs. confidence language, and tone shifts vs. the
  prior period.
- **Risk factor tracking** (`app/extraction/risks.py`): extracts categorized
  risk factors and flags new, escalated, and removed risks vs. the prior
  period.
- **Competitor benchmarking** (`app/benchmarking/`): cross-company table of
  revenue, growth, margins, capex intensity, and leverage.
- **Investment memo generation** (`app/memo/`): company overview, financial
  summary, bull case, bear case, key risks, and due-diligence questions --
  grounded entirely in the extracted data above.
- **FastAPI app** (`app/main.py`, `app/api/routes.py`): REST endpoints for
  every module above, plus a free-form RAG Q&A endpoint.

## Architecture notes

- **Hybrid extraction.** Clean tabular data (financial statements, risk
  factor headers) is parsed deterministically with regex for 100% accuracy.
  Narrative data (forward guidance figures, management tone, memo prose) uses
  small, focused LLM calls. This combination is what makes the >90-95%
  accuracy targets achievable with a free-tier LLM.
- **Deterministic comparisons.** Period-over-period deltas (financial metric
  changes, tone shifts, new/escalated/removed risks) are computed in plain
  Python from already-extracted structured data, not via the LLM.
- **LLM provider.** Chat completions go through
  [OpenRouter](https://openrouter.ai/) (OpenAI-compatible API) so the project
  can run on free-tier models. Embeddings run locally via
  sentence-transformers, so the RAG pipeline has no embedding-API dependency.
- **Caching.** `app/state.py` caches parsed documents, the vector store, and
  LLM-backed extraction results (financial metrics, tone, risk factors) per
  ticker/period for the life of the process, to minimize redundant calls
  against free-tier rate limits.

## Setup

Requires Python 3.12 (chromadb's dependencies are incompatible with 3.14).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key (free tier available at
https://openrouter.ai/keys). The default model
(`nvidia/nemotron-nano-9b-v2:free`) is a free-tier model; OpenRouter free
models are subject to a shared daily rate limit (50 requests/day without
adding credits, 1000/day with $10+ in account credits).

## Sample data

`data/sample_filings/` contains two synthetic companies (SOLR, VCLD), each
with a FY2023 and FY2024 10-K (PDF) and earnings call transcript (.txt), plus
`ground_truth.json` with the expected extraction values used by the test
suite. Regenerate with:

```bash
python scripts/generate_sample_data.py
```

## Running the API

```bash
uvicorn app.main:app --reload
```

Then visit http://127.0.0.1:8000/docs for interactive API docs.

### Dashboard (frontend)

Open http://127.0.0.1:8000/ for the built-in dashboard — a single-page app
(served from `app/static/`) that lets you:

- **Upload a filing**: drag & drop a 10-K PDF or earnings-call transcript
  (`.txt`), enter the ticker/fiscal year/document type, and it's parsed,
  indexed into the RAG vector store, and immediately selectable in the
  company/period dropdowns.
- **Browse a company/period** via the Company and Period selectors at the
  top, then switch between tabs:
  - **Metrics** — extracted financial metrics + forward guidance
  - **YoY Comparison** — period-over-period deltas
  - **Management Tone** — sentiment, confidence score, hedging/confidence
    quotes, and tone shift vs. the prior period
  - **Risk Factors** — categorized risks with new/escalated/removed tags
    vs. the prior period
  - **Benchmark** — pick peer tickers and build a cross-company comparison
    table + chart
  - **Investment Memo** — LLM-generated memo (overview, financial summary,
    bull/bear case, key risks, open questions); optionally pick peers for
    benchmarking context
  - **Ask the Filings** — free-form RAG chat over the indexed documents

The UI is plain HTML/CSS/JS (no build step) with a dark glassmorphism theme
and Chart.js for the benchmark visualization.

### Key endpoints

| Endpoint | Description |
| --- | --- |
| `GET /api/companies` | List all ingested companies/periods |
| `GET /api/metrics/{ticker}/{period}` | Financial metrics (revenue, margins, cash flow, guidance) |
| `GET /api/metrics/{ticker}/{period}/comparison` | YoY/QoQ metric deltas |
| `GET /api/tone/{ticker}/{period}` | Management tone analysis |
| `GET /api/tone/{ticker}/{period}/comparison` | Tone shift vs. prior period |
| `GET /api/risks/{ticker}/{period}` | Categorized risk factors |
| `GET /api/risks/{ticker}/{period}/comparison` | New/escalated/removed risks vs. prior period |
| `GET /api/benchmark/{period}?tickers=SOLR,VCLD` | Cross-company benchmark table |
| `GET /api/memo/{ticker}/{period}?peers=VCLD` | Generated investment memo |
| `POST /api/rag/query` | Free-form RAG Q&A over the filings |
| `POST /api/upload` | Upload a 10-K PDF or earnings-call transcript (multipart: `file`, `ticker`, `fiscal_year`, `doc_type`); parses, indexes into RAG, and makes it available everywhere else |

## Tests

```bash
pytest tests/ -v
```

Tests validate each success metric against `ground_truth.json`:

- Financial metric extraction accuracy (deterministic, always runs)
- Risk factor extraction and new/escalated risk flagging (deterministic, always runs)
- Benchmark table arithmetic (deterministic, always runs)
- Forward guidance extraction, tone discrimination, and memo grounding
  (require a live LLM call -- automatically skipped if the free-tier
  provider is currently rate-limited)

## Project layout

```
app/
  ingestion/      # PDF/txt parsing into structured ParsedDocument sections
  rag/             # chunking, embeddings, Chroma vector store, retrieval
  extraction/      # financial metrics, YoY/QoQ comparison, risk factors
  analysis/        # management tone analysis
  benchmarking/    # cross-company benchmark table
  memo/            # investment memo generation
  api/             # FastAPI routes
  models/          # Pydantic schemas shared across modules
  state.py         # cached documents, vector store, extraction results
  config.py        # environment/config
  main.py          # FastAPI app entrypoint
  static/          # frontend dashboard (HTML/CSS/JS, served at /)
data/sample_filings/  # synthetic test filings + ground_truth.json
scripts/              # sample data generation
tests/                # pytest suite validating accuracy success metrics
```
