from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..dataloader import load_records
from ..parser import extract_entities
from ..io.qdrant import QdrantIO
from ..io.neo4j import Neo4jIO
from ..retrievals.qdrant import QdrantRetriever
from ..retrievals.neo4j import Neo4jRetriever
from .metrics import (
    BackendMetrics,
    PerQueryRecord,
    percentile,
    recall_at_k,
    mrr_at_k,
    find_rank,
)


def run_index(backends: List[str], data_path: Path) -> Dict[str, Any]:
    cfg = Config()
    res: Dict[str, Any] = {}

    if "qdrant" in backends:
        t0 = time.perf_counter()
        qio = QdrantIO(cfg)
        idx = qio.index_dataset(data_path=data_path)
        res["qdrant"] = {
            "indexed": idx.get("indexed", 0),
            "batches": idx.get("batches", 0),
            "collection": idx.get("collection") or idx.get("base_collection"),
            "per_format_counts": idx.get("per_format_counts", {}),
            "vector_size": idx.get("vector_size"),
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    if "neo4j" in backends:
        n4j = Neo4jIO(cfg, use_read_only=False)
        try:
            t0 = time.perf_counter()
            stats = n4j.ingest(data_path=data_path)
            res["neo4j"] = {
                "documents": stats.documents,
                "entities": stats.entities,
                "mentions": stats.mentions,
                "refers_to": stats.refers_to,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            }
        finally:
            n4j.close()

    return res


def _compute_backend_metrics(
    records: List[Dict[str, Any]],
    backends: List[str],
    top_k: int,
    no_correction: bool,
    cfg: Config,
    restrict_format: Optional[str],
    restrict_formats: Optional[List[str]] = None,
) -> Tuple[Dict[str, BackendMetrics], Dict[str, List[Dict[str, Any]]]]:
    local_summary: Dict[str, BackendMetrics] = {}
    local_per_query: Dict[str, List[Dict[str, Any]]] = {}

    if "qdrant" in backends:
        retr = QdrantRetriever(cfg)
        latencies: List[float] = []
        recalls: List[float] = []
        mrrs: List[float] = []
        pqs: List[Dict[str, Any]] = []
        corr_applied = 0
        corr_helped = 0

        for rec in records:
            q = rec["question"]
            rid = rec["id"]

            t0 = time.perf_counter()
            hits = retr.search(
                query=q,
                top_k=top_k,
                source_format=restrict_format,
                source_formats=restrict_formats,
            )
            lat = (time.perf_counter() - t0) * 1000.0

            r = recall_at_k(hits, rid, top_k)
            m = mrr_at_k(hits, rid, top_k)

            applied = False
            helped = False

            if not no_correction and cfg.enable_self_correction and r == 0.0:
                qents = extract_entities(q)
                if qents:
                    applied = True
                    corr_applied += 1
                    aug = f"{q} " + " ".join(qents)
                    t1 = time.perf_counter()
                    hits2 = retr.search(
                        query=aug,
                        top_k=top_k,
                        source_format=restrict_format,
                        source_formats=restrict_formats,
                    )
                    lat2 = (time.perf_counter() - t1) * 1000.0
                    r2 = recall_at_k(hits2, rid, top_k)
                    m2 = mrr_at_k(hits2, rid, top_k)
                    if (r2 > r) or (r2 == r and m2 > m):
                        helped = r2 > r
                        hits = hits2
                        lat = lat2
                        r = r2
                        m = m2
                        if helped:
                            corr_helped += 1

            latencies.append(lat)
            recalls.append(r)
            mrrs.append(m)

            rank, hit = find_rank(hits, rid)
            pqs.append(
                asdict(
                    PerQueryRecord(
                        id=rid,
                        rank=rank,
                        score=(hit.score if hit else 0.0),
                        latency_ms=lat,
                        correction_applied=applied,
                        components=(hit.components if hit else {}),
                    )
                )
            )

        local_summary["qdrant"] = BackendMetrics(
            recall_at_5=float(sum(recalls)) / float(len(recalls) or 1),
            mrr_at_5=float(sum(mrrs)) / float(len(mrrs) or 1),
            query_latency_ms_mean=float(sum(latencies)) / float(len(latencies) or 1),
            query_latency_ms_p95=percentile(latencies, 0.95),
            index_time_ms=0.0,
            corrections_applied=corr_applied,
            corrections_helped=corr_helped,
        )
        local_per_query["qdrant"] = pqs

    if "neo4j" in backends:
        retr = Neo4jRetriever(cfg)
        try:
            latencies_n: List[float] = []
            recalls_n: List[float] = []
            mrrs_n: List[float] = []
            pqs_n: List[Dict[str, Any]] = []
            corr_applied = 0
            corr_helped = 0

            for rec in records:
                q = rec["question"]
                rid = rec["id"]

                t0 = time.perf_counter()
                hits = retr.search(
                    query=q,
                    top_k=top_k,
                    source_format=restrict_format,
                    source_formats=restrict_formats,
                )
                lat = (time.perf_counter() - t0) * 1000.0

                r = recall_at_k(hits, rid, top_k)
                m = mrr_at_k(hits, rid, top_k)

                applied = False
                helped = False

                if not no_correction and cfg.enable_self_correction and r == 0.0:
                    qents = extract_entities(q)
                    if qents:
                        applied = True
                        corr_applied += 1
                        aug = f"{q} " + " ".join(qents)
                        t1 = time.perf_counter()
                        hits2 = retr.search(
                            query=aug,
                            top_k=top_k,
                            source_format=restrict_format,
                            source_formats=restrict_formats,
                        )
                        lat2 = (time.perf_counter() - t1) * 1000.0
                        r2 = recall_at_k(hits2, rid, top_k)
                        m2 = mrr_at_k(hits2, rid, top_k)
                        if (r2 > r) or (r2 == r and m2 > m):
                            helped = r2 > r
                            hits = hits2
                            lat = lat2
                            r = r2
                            m = m2
                            if helped:
                                corr_helped += 1

                latencies_n.append(lat)
                recalls_n.append(r)
                mrrs_n.append(m)

                rank, hit = find_rank(hits, rid)
                pqs_n.append(
                    asdict(
                        PerQueryRecord(
                            id=rid,
                            rank=rank,
                            score=(hit.score if hit else 0.0),
                            latency_ms=lat,
                            correction_applied=applied,
                            components=(hit.components if hit else {}),
                        )
                    )
                )

            local_summary["neo4j"] = BackendMetrics(
                recall_at_5=float(sum(recalls_n)) / float(len(recalls_n) or 1),
                mrr_at_5=float(sum(mrrs_n)) / float(len(mrrs_n) or 1),
                query_latency_ms_mean=float(sum(latencies_n))
                / float(len(latencies_n) or 1),
                query_latency_ms_p95=percentile(latencies_n, 0.95),
                index_time_ms=0.0,
                corrections_applied=corr_applied,
                corrections_helped=corr_helped,
            )
            local_per_query["neo4j"] = pqs_n
        finally:
            retr.close()

    return local_summary, local_per_query


def run_benchmark(
    backends: List[str],
    data_path: Path,
    out_path: Path,
    top_k_override: Optional[int] = None,
    no_correction: bool = False,
    formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = Config()
    top_k = top_k_override or cfg.top_k

    data = load_records(data_path)
    # If formats are specified, restrict the dataset to those formats for combined metrics
    if formats:
        allowed = {f.strip().lower() for f in formats if f.strip()}
        data = [r for r in data if str(r.get("source_format", "")).lower() in allowed]

    summary: Dict[str, BackendMetrics] = {}
    per_query: Dict[str, List[Dict[str, Any]]] = {}
    by_format_summary: Dict[str, Dict[str, BackendMetrics]] = {}
    by_format_per_query: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    # Combined metrics across selected formats
    s_all, pq_all = _compute_backend_metrics(
        data,
        backends,
        top_k,
        no_correction,
        cfg,
        None,
        restrict_formats=[str(r.get("source_format")) for r in data if r.get("source_format")],
    )
    summary.update(s_all)
    per_query.update(pq_all)

    # Per-format (optional)
    valid_fmts = {"json", "txt", "xml"}
    if formats:
        for fmt in formats:
            sf = fmt.lower().strip()
            if sf not in valid_fmts:
                continue
            fdata = [r for r in data if r.get("source_format") == sf]
            s_fmt, pq_fmt = _compute_backend_metrics(
                fdata,
                backends,
                top_k,
                no_correction,
                cfg,
                sf,
                restrict_formats=[sf],
            )
            by_format_summary[sf] = s_fmt
            by_format_per_query[sf] = pq_fmt

    result = {
        "config": {
            "model": cfg.ollama_model,
            "top_k": top_k,
            "weights": {"alpha": cfg.alpha, "beta": cfg.beta, "gamma": cfg.gamma},
            "enable_self_correction": cfg.enable_self_correction and not no_correction,
            "backends": backends,
        },
        "summary": {k: asdict(v) for k, v in summary.items()},
        "per_query": per_query,
    }

    if by_format_summary:
        result["by_format"] = {
            f: {k: asdict(v) for k, v in s.items()}
            for f, s in by_format_summary.items()
        }
        result["by_format_per_query"] = by_format_per_query

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
