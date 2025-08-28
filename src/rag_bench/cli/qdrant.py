from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ..config import Config
from ..io.qdrant import QdrantIO


def _cli_index(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    cfg = Config()
    qio = QdrantIO(cfg)
    summary = qio.index_dataset(
        data_path=data_path, batch_size=args.batch_size, warmup=not args.no_warmup
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _cli_search(args: argparse.Namespace) -> None:
    cfg = Config()
    qio = QdrantIO(cfg)
    hits = qio.search(
        query=args.query, top_k=args.top_k, source_format=getattr(args, "format", None)
    )
    print(json.dumps({"count": len(hits), "hits": hits}, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Qdrant indexing and search utilities."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser(
        "index", help="Index records from a JSON Lines dataset into Qdrant."
    )
    p_index.add_argument(
        "--data",
        "-d",
        default=str(Path("data") / "qa.json"),
        help="Path to dataset file or directory (json/jsonl/ndjson/txt/xml)",
    )
    p_index.add_argument("--batch-size", type=int, default=32, help="Upsert batch size")
    p_index.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup; still performs one embed to derive dim",
    )
    p_index.set_defaults(func=_cli_index)

    p_search = sub.add_parser(
        "search", help="Query top-k nearest neighbors from Qdrant."
    )
    p_search.add_argument(
        "--query", "-q", required=True, help="Query text to embed and search"
    )
    p_search.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of results (defaults to config TOP_K)",
    )
    p_search.add_argument(
        "--format",
        choices=["json", "txt", "xml"],
        default=None,
        help="Restrict search to a single source format",
    )
    p_search.set_defaults(func=_cli_search)

    args = parser.parse_args(argv)
    args.func(args)
