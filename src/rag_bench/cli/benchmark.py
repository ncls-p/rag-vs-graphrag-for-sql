from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ..config import Config
from ..bench.runner import run_index, run_benchmark


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RAG backends (Qdrant vs Neo4j)"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run quick health checks (Ollama, Qdrant, Neo4j) and exit",
    )
    parser.add_argument(
        "--stage",
        choices=["index", "run", "all"],
        default="all",
        help="Which stage to execute",
    )
    parser.add_argument(
        "--backends",
        default="qdrant,neo4j",
        help="Comma-separated backends: qdrant,neo4j",
    )
    parser.add_argument(
        "--data",
        "-d",
        default=str(Path("data") / "qa.json"),
        help="Path to dataset file or directory (json/jsonl/ndjson/txt/xml)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="benchmark_results.json",
        help="Where to write benchmark results (for run/all)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override TOP_K")
    parser.add_argument(
        "--no-self-correction", action="store_true", help="Disable self-correction loop"
    )
    parser.add_argument(
        "--formats",
        default=None,
        help="Comma-separated list of formats to isolate (e.g., 'json,txt,xml')",
    )

    args = parser.parse_args(argv)

    if args.health:
        cfg = Config()
        ok_oll, msg_oll = cfg.health_ollama()
        ok_qd, msg_qd = cfg.health_qdrant()
        ok_n4j, msg_n4j = cfg.health_neo4j()
        print(
            json.dumps(
                {
                    "ollama": {"ok": ok_oll, "msg": msg_oll},
                    "qdrant": {"ok": ok_qd, "msg": msg_qd},
                    "neo4j": {"ok": ok_n4j, "msg": msg_n4j},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    data_path = Path(args.data)
    out_path = Path(args.out)
    backends: List[str] = [
        b.strip().lower() for b in args.backends.split(",") if b.strip()
    ]

    if args.stage in ("index", "all"):
        idx_res = run_index(backends, data_path)
        print(
            json.dumps(
                {"stage": "index", "result": idx_res}, ensure_ascii=False, indent=2
            )
        )

    if args.stage in ("run", "all"):
        fmts = None
        if args.formats:
            fmts = [s.strip().lower() for s in args.formats.split(",") if s.strip()]
        res = run_benchmark(
            backends,
            data_path,
            out_path,
            top_k_override=args.top_k,
            no_correction=bool(args.no_self_correction),
            formats=fmts,
        )
        print(json.dumps({"stage": "run", "result": res}, ensure_ascii=False, indent=2))
