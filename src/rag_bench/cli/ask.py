from __future__ import annotations

import argparse
import json
from typing import Optional

from ..rag import answer_question, generate_search_query


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="RAG question answering using OpenAI-compatible API"
    )
    p.add_argument("--query", "-q", required=True, help="Question to answer")
    p.add_argument("--backend", choices=["qdrant", "neo4j"], default="qdrant")
    p.add_argument("--top-k", "-k", type=int, default=20)
    p.add_argument("--format", choices=["json", "txt", "xml"], default=None)
    p.add_argument(
        "--gen-instruction",
        default=None,
        help="Custom instruction to guide query generation",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args(argv)

    # 1) Ask LLM to generate a focused retrieval query
    retrieval_query = generate_search_query(
        question=args.query,
        backend=args.backend,
        source_format=args.format,
        instruction=args.gen_instruction,
    )

    # 2) Retrieve K=20 docs and answer with RAG
    res = answer_question(
        question=args.query,
        backend=args.backend,
        top_k=args.top_k or 20,
        source_format=args.format,
        retrieval_query=retrieval_query,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {
                "backend": res.backend,
                "model": res.model,
                "question": res.question,
                "retrieval_query": res.retrieval_query,
                "answer": res.answer,
                "contexts": res.contexts,
                "usage": res.usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
