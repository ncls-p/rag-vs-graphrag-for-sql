from __future__ import annotations

import argparse
import json
from typing import Optional

from ..retrievals import QdrantRetriever, Neo4jRetriever


def _cli_qdrant(args: argparse.Namespace) -> None:
    retr = QdrantRetriever()
    hits = retr.search(
        query=args.query, top_k=args.top_k, source_format=getattr(args, "format", None)
    )
    print(
        json.dumps(
            {
                "backend": "qdrant",
                "count": len(hits),
                "hits": [
                    {
                        "id": h.id,
                        "score": h.score,
                        "components": h.components,
                        "payload": h.payload,
                    }
                    for h in hits
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _cli_neo4j(args: argparse.Namespace) -> None:
    retr = Neo4jRetriever()
    try:
        hits = retr.search(
            query=args.query,
            top_k=args.top_k,
            source_format=getattr(args, "format", None),
        )
        print(
            json.dumps(
                {
                    "backend": "neo4j",
                    "count": len(hits),
                    "hits": [
                        {
                            "id": h.id,
                            "score": h.score,
                            "components": h.components,
                            "payload": h.payload,
                        }
                        for h in hits
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        retr.close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Unified retrieval for Qdrant and Neo4j."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_q = sub.add_parser("qdrant", help="Query Qdrant backend")
    p_q.add_argument("--query", "-q", required=True, help="Query text")
    p_q.add_argument("--top-k", "-k", type=int, default=None, help="Top-K")
    p_q.add_argument(
        "--format",
        choices=["json", "txt", "xml"],
        default=None,
        help="Restrict search to a single source format",
    )
    p_q.set_defaults(func=_cli_qdrant)

    p_n = sub.add_parser("neo4j", help="Query Neo4j backend with composite scoring")
    p_n.add_argument("--query", "-q", required=True, help="Query text")
    p_n.add_argument("--top-k", "-k", type=int, default=None, help="Top-K")
    p_n.add_argument(
        "--format",
        choices=["json", "txt", "xml"],
        default=None,
        help="Restrict search to a single source format",
    )
    p_n.set_defaults(func=_cli_neo4j)

    args = parser.parse_args(argv)
    args.func(args)
