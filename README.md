# rag-bench

Benchmark and compare Retrieval-Augmented Generation (RAG) backends for SQL-oriented knowledge. It evaluates a vector database (Qdrant) versus a graph database (Neo4j) on the same corpus, with strict isolation across data formats (JSON/TXT/XML).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture & Workflow](#architecture--workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run Services (Docker)](#run-services-docker)
- [Configuration](#configuration)
- [Quickstart](#quickstart)
- [CLI Reference](#cli-reference)
- [Data Formats](#data-formats)
- [Retrieval Details](#retrieval-details)
- [Benchmark Output](#benchmark-output)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Overview
rag-bench indexes question/answer pairs (including DDL/SQL snippets), runs retrieval over Qdrant and Neo4j, and reports quality and latency metrics. An Ollama-compatible embedding service provides the embeddings.

## Features
- Dual backends: Qdrant (semantic ANN) and Neo4j (semantic + entity + neighbor signals)
- Format isolation: JSON/TXT/XML stored and evaluated independently
- Self-correction: retries with entity-augmented queries when needed
- Clean CLIs: health checks, indexing, benchmarking, ingesting, searching, stats

## Architecture & Workflow
1. Load & normalize data
   - Inputs: .json/.jsonl/.ndjson, .txt, .xml (file or directory)
   - Each record → id, question, answer_text, entities, doc_type, tags, source_format
   - Derived folders starting with `output` are ignored
2. Index
   - Qdrant: per-format collections `<QDRANT_COLLECTION>_{json,txt,xml}`
   - Neo4j: Document nodes (+ embedding, entities, doc_type, source_format), Entity nodes, MENTIONS edges, REFERS_TO edges between docs that share entities
3. Retrieve
   - Qdrant: nearest-neighbor over embeddings (format restriction supported)
   - Neo4j: composite score = alpha*semantic + beta*entity_jaccard + gamma*neighbor_boost
4. Benchmark
   - Compute Recall@K, MRR@K, mean and p95 latency; optional self-correction pass
   - Output JSON includes combined metrics and per-format breakdown

## Requirements
- Python 3.13+
- Optional: Docker (to run Qdrant/Neo4j locally)

## Installation
Recommended (console scripts):
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```
Alternative (direct packages):
```bash
python -m pip install requests pydantic tenacity numpy tqdm qdrant-client neo4j
```

## Run Services (Docker)
```bash
docker compose up -d
```
- Qdrant: http://localhost:6333 (HTTP)
- Neo4j: http://localhost:7474 (HTTP), bolt://localhost:7687 (Bolt)
- Default Neo4j credentials (from compose): `neo4j / neo4jtest`

## Configuration
Environment variables (defaults in `src/rag_bench/config.py`):

| Variable | Default | Description |
|---------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434 | Base URL for embedding server |
| OLLAMA_EMBED_MODEL | dengcao/Qwen3-Embedding-0.6B:q8_0 | Embedding model name |
| HTTP_TIMEOUT_SECONDS | 30 | HTTP timeout per request |
| REQUEST_RETRIES | 3 | HTTP retry count |
| QDRANT_URL | http://localhost:6333 | Qdrant base URL |
| QDRANT_API_KEY | – | Qdrant API key (optional) |
| QDRANT_COLLECTION | qa_demo | Base collection name |
| NEO4J_URI | bolt://localhost:7687 | Neo4j Bolt URL |
| NEO4J_USER / NEO4J_PASSWORD | – | Credentials for writes |
| NEO4J_RO_USER / NEO4J_RO_PASSWORD | – | Read-only credentials (optional) |
| TOP_K | 5 | Number of hits |
| SHORTLIST_SIZE | 50 | Neo4j shortlist size for reranking |
| NEO4J_SCORE_ALPHA | 0.7 | Semantic weight |
| NEO4J_SCORE_BETA | 0.2 | Entity Jaccard weight |
| NEO4J_SCORE_GAMMA | 0.1 | Neighbor boost weight |
| ENABLE_SELF_CORRECTION | true | Enable second-pass augmentation |
| ALLOW_DESTRUCTIVE_OPS | false | Allow destructive Neo4j ops |

Embedding endpoint test (one of these should work):
```bash
curl -sS -X POST http://localhost:11434/api/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"dengcao/Qwen3-Embedding-0.6B:q8_0","input":"hello"}'
curl -sS -X POST http://localhost:11434/api/embed -H 'Content-Type: application/json' \
  -d '{"model":"dengcao/Qwen3-Embedding-0.6B:q8_0","input":"hello"}'
```

## Quickstart
```bash
# 1) Start services (optional if hosted)
docker compose up -d

# 2) Configure creds if needed
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jtest

# 3) Health check
rag-benchmark --health

# 4) Index both backends
rag-benchmark --stage index --backends qdrant,neo4j --data data/qa.json

# 5) Run benchmark and write report
rag-benchmark --stage run --backends qdrant,neo4j --data data/qa.json --out output/benchmark_results.json

# 6) Per-format comparison (isolation)
rag-benchmark --stage run --backends qdrant,neo4j --data data --formats json,txt,xml --out output/benchmark_by_format.json
```

## CLI Reference
Console scripts installed by `-e .`:

### rag-benchmark
```bash
# health
rag-benchmark --health
# index
rag-benchmark --stage index --backends qdrant,neo4j --data data/qa.json
# run
rag-benchmark --stage run --backends qdrant,neo4j --data data/qa.json --out output/benchmark_results.json
```

### rag-retrieval
```bash
# Qdrant
rag-retrieval qdrant --query "Where is swaping activity?" --top-k 5 --format json
# Neo4j
rag-retrieval neo4j --query "Which table provides CICS transaction stats?" --top-k 5 --format json
```

### rag-qdrant
```bash
rag-qdrant index --data data/qa.json
rag-qdrant search --query "top Db2 packages" --top-k 5 --format json
```

### rag-neo4j
```bash
rag-neo4j ingest --data data/qa.json
rag-neo4j stats
ALLOW_DESTRUCTIVE_OPS=true rag-neo4j drop
```

Module invocation (alternative):
```bash
PYTHONPATH=src python -m rag_bench.benchmark --stage all
PYTHONPATH=src python -m rag_bench.retrieval qdrant --query "..."
```

## Data Formats
- JSON/JSONL/NDJSON: objects or line-delimited; nested arrays supported
- TXT: "qa.txt" style with `@` question markers; fallback uses first non-empty line as question
- XML: best-effort search for `<question>`/`<answer>`; otherwise treat content as text
- Each record gets `source_format` (json/txt/xml) and an inferred `doc_type` (DDL, SQL_QUERY, or PLAIN)

## Retrieval Details
- Qdrant: cosine distance search over embeddings; returns payloads
- Neo4j scoring: `score = alpha*semantic + beta*entity_jaccard + gamma*neighbor_boost`
  - `neighbor_boost` is the best semantic score among REFERS_TO neighbors
  - Entities are extracted via an uppercase token heuristic filtering common SQL keywords

## Benchmark Output
Result JSON contains:
- `config`: model, top_k, weights, backends, self-correction
- `summary`: per-backend metrics (recall@5, mrr@5, mean/p95 latency)
- `per_query`: rank, score, latency, components per question
- `by_format` and `by_format_per_query` when `--formats` is supplied

## Troubleshooting
- Embeddings: verify `OLLAMA_BASE_URL` and model; try both `/api/embeddings` and `/api/embed`
- Qdrant: check `QDRANT_URL` and container health (`curl http://localhost:6333/`)
- Neo4j auth: ensure `NEO4J_USER/NEO4J_PASSWORD` match your instance
- Vector size mismatch in Qdrant: recreate the collection or change `QDRANT_COLLECTION`
- Ensure derived outputs live under `output*/` so the loader ignores them

## Development
- Modular layout: `cli/`, `io/`, `retrievals/`, `bench/`, `utils/`
- You can also run CLIs via modules:
  - `PYTHONPATH=src python -m rag_bench.benchmark --stage all`
  - `PYTHONPATH=src python -m rag_bench.retrieval qdrant --query "..."`

## License
No explicit license is included. Add one if you plan to redistribute.
