from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..rag import answer_question, generate_search_query


def _parse_list(val: Optional[str], defaults: Optional[List[str]] = None) -> List[str]:
    if not val:
        return list(defaults or [])
    if isinstance(val, list):
        return list(val)
    parts = [p.strip() for p in str(val).split(",") if p.strip()]
    return parts or list(defaults or [])


def _iter_questions(path: Path) -> Iterable[str]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    # JSON array or object
    if text.startswith("[") or text.startswith("{"):
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and isinstance(item.get("question"), str):
                    yield item["question"].strip()
                elif isinstance(item, str) and item.strip():
                    yield item.strip()
            return
        if isinstance(obj, dict):
            # support {"records": [...]}
            if isinstance(obj.get("records"), list):
                for item in obj["records"]:
                    if isinstance(item, dict) and isinstance(item.get("question"), str):
                        yield item["question"].strip()
                    elif isinstance(item, str) and item.strip():
                        yield item.strip()
                return
            # single object with question
            if isinstance(obj.get("question"), str):
                yield obj["question"].strip()
                return
    # JSON Lines (qa.json style) or plain text lines
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            yield s
            continue
        if isinstance(obj, dict) and isinstance(obj.get("question"), str):
            yield obj["question"].strip()
        elif isinstance(obj, str) and obj.strip():
            yield obj.strip()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Batch RAG ask from a questions file")
    p.add_argument("--in", dest="in_path", required=True, help="Input questions file (JSON/JSONL)")
    p.add_argument("--out", dest="out_path", default=str(Path("output") / "batch_answers.json"), help="Output JSON file")
    p.add_argument("--backends", default="qdrant,neo4j", help="Comma-separated backends (qdrant,neo4j)")
    p.add_argument(
        "--formats",
        default="json,txt,xml",
        help="Comma-separated formats to restrict (json,txt,xml). Use 'all' for no restriction",
    )
    p.add_argument("--top-k", "-k", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--gen-instruction", default=None, help="Custom instruction for query generation")
    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = list(_iter_questions(in_path))
    if not questions:
        print(json.dumps({"error": "no questions found", "input": str(in_path)}))
        return

    backends = _parse_list(args.backends, defaults=["qdrant", "neo4j"])
    fmts_in = _parse_list(args.formats, defaults=["json", "txt", "xml"])
    # "all" means no restriction
    fmts: List[Optional[str]] = [None] if any(f.lower() == "all" for f in fmts_in) else fmts_in  # type: ignore[list-item]

    results: List[Dict[str, Any]] = []
    for q in questions:
        for bk in backends:
            for sf in fmts:
                try:
                    rq = generate_search_query(
                        question=q,
                        backend=bk,
                        source_format=sf if isinstance(sf, str) else None,
                        instruction=args.gen_instruction,
                    )
                except Exception as e:
                    results.append(
                        {
                            "question": q,
                            "backend": bk,
                            "format": sf,
                            "error": f"query_generation_failed: {e}",
                        }
                    )
                    continue

                try:
                    ans = answer_question(
                        question=q,
                        backend=bk,
                        top_k=args.top_k,
                        source_format=sf if isinstance(sf, str) else None,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        retrieval_query=rq,
                    )
                except Exception as e:
                    results.append(
                        {
                            "question": q,
                            "backend": bk,
                            "format": sf,
                            "retrieval_query": rq,
                            "error": f"answer_failed: {e}",
                        }
                    )
                    continue

                # contexts already contain id, score, content and possibly source_format
                results.append(
                    {
                        "backend": ans.backend,
                        "format": sf,
                        "model": ans.model,
                        "question": ans.question,
                        "retrieval_query": ans.retrieval_query,
                        "answer": ans.answer,
                        "contexts": ans.contexts,
                        "usage": ans.usage,
                    }
                )

    # Save as an array JSON
    out_path.write_text(json.dumps({"count": len(results), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "count": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
