from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ..config import Config
from ..io.neo4j import Neo4jIO


def _cli_ingest(args: argparse.Namespace) -> None:
    cfg = Config()
    n4j = Neo4jIO(cfg, use_read_only=False)
    try:
        stats = n4j.ingest(
            data_path=Path(args.data),
            batch_size=args.batch_size,
            create_refers_to=not args.no_refers,
        )
        print(
            json.dumps(
                {
                    "documents": stats.documents,
                    "entities": stats.entities,
                    "mentions": stats.mentions,
                    "refers_to": stats.refers_to,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        n4j.close()


def _cli_stats(args: argparse.Namespace) -> None:
    cfg = Config()
    n4j = Neo4jIO(cfg, use_read_only=True)
    try:
        print(json.dumps(n4j.stats(), ensure_ascii=False, indent=2))
    finally:
        n4j.close()


def _cli_drop(args: argparse.Namespace) -> None:
    cfg = Config()
    n4j = Neo4jIO(cfg, use_read_only=False)
    try:
        res = n4j.drop_graph()
        print(json.dumps(res, ensure_ascii=False, indent=2))
    finally:
        n4j.close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Neo4j ingestion and utilities for RAG graph."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser(
        "ingest",
        help="Ingest dataset into Neo4j with embeddings, MENTIONS, REFERS_TO.",
    )
    p_ing.add_argument(
        "--data",
        "-d",
        default=str(Path("data") / "qa.json"),
        help="Path to dataset file or directory (json/jsonl/ndjson/txt/xml)",
    )
    p_ing.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (reserved; embedding is per doc)",
    )
    p_ing.add_argument(
        "--no-refers", action="store_true", help="Skip creation of REFERS_TO edges"
    )
    p_ing.set_defaults(func=_cli_ingest)

    p_stats = sub.add_parser(
        "stats", help="Print basic counts of nodes and relationships."
    )
    p_stats.set_defaults(func=_cli_stats)

    p_drop = sub.add_parser(
        "drop",
        help="Delete all nodes and relationships (requires ALLOW_DESTRUCTIVE_OPS=true).",
    )
    p_drop.set_defaults(func=_cli_drop)

    args = parser.parse_args(argv)
    args.func(args)
