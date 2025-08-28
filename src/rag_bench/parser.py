import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

QUESTION_PREFIX = "@"
AGENT_MARKER = "Agent restructured data"

SQL_STOPWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "CREATE",
    "TABLE",
    "PRIMARY",
    "KEY",
    "NOT",
    "NULL",
    "DEFAULT",
    "INDEX",
    "UNIQUE",
    "ON",
    "IN",
    "BY",
    "ROW",
    "ASC",
    "DESC",
    "DATE",
    "TIME",
    "FLOAT",
    "CHAR",
    "INT",
    "AND",
    "OR",
    "AS",
    "VALUES",
    "SET",
    "WITH",
    "ORDER",
    "GROUP",
    "HAVING",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "INSERT",
    "UPDATE",
    "DELETE",
    "FOREIGN",
}

DOC_TYPES = ("PLAIN", "DDL", "SQL_QUERY")


def is_question_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return s.startswith(QUESTION_PREFIX)


def extract_question_text(line: str) -> str:
    s = line.strip()
    # Remove leading '@' and an optional token (e.g., '@DRLZ1')
    if s.startswith("@"):
        s = s[1:].lstrip()
    # If agent marker exists, use the text after it
    marker_idx = s.find(AGENT_MARKER)
    if marker_idx != -1:
        q = s[marker_idx + len(AGENT_MARKER) :].strip()
        return q if q else s
    # Else drop leading token until first space
    parts = s.split(None, 1)
    if len(parts) == 2:
        return parts[1].strip()
    return s


def detect_doc_type(text: str) -> str:
    t = text.upper()
    if any(
        kw in t for kw in ("CREATE TABLE", "ALTER TABLE", "PRIMARY KEY", "FOREIGN KEY")
    ):
        return "DDL"
    if any(kw in t for kw in ("SELECT ", "WITH ", "INSERT ", "UPDATE ", "DELETE ")):
        return "SQL_QUERY"
    return "PLAIN"


ENTITY_TOKEN_RE = re.compile(r"[A-Z][A-Z0-9_.]{2,}")


def normalize_entity_token(tok: str) -> str:
    tok = tok.strip().strip("\\\"'`,;()")
    # Remove trailing punctuation again after stripping quotes
    tok = tok.strip('",;()')
    return tok


def extract_entities(text: str, min_len: int = 3) -> List[str]:
    candidates = ENTITY_TOKEN_RE.findall(text)
    out = []
    seen = set()
    for raw in candidates:
        norm = normalize_entity_token(raw)
        if not norm:
            continue
        if len(norm) < min_len:
            continue
        if norm in SQL_STOPWORDS:
            continue
        # Filter obvious types like FLOAT(8) captured as FLOAT
        if norm.isalpha() and norm in SQL_STOPWORDS:
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def iter_records(lines: Iterable[str]) -> Iterable[Dict]:
    current_q: str | None = None
    current_ans: List[str] = []

    def flush(acc: List[Dict], q: str, ans_lines: List[str]):
        answer_lines = [line.strip() for line in ans_lines if line.strip()]
        answer_text = "\n".join(answer_lines)
        combined = (q + "\n" + answer_text).strip()
        doc_type = detect_doc_type(combined)
        entities = extract_entities(combined)
        tags = [doc_type]
        if entities:
            tags.append("has_entities")
        acc.append(
            {
                "question": q,
                "answer_text": answer_text,
                "answer_lines": answer_lines,
                "tags": tags,
                "entities": entities,
                "doc_type": doc_type,
            }
        )

    acc: List[Dict] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if is_question_line(line):
            q_text = extract_question_text(line)
            if current_q is not None:
                flush(acc, current_q, current_ans)
                current_ans = []
            current_q = q_text
        else:
            # Accumulate non-empty answer lines
            if current_q is not None and line.strip():
                current_ans.append(line.strip())

    # Flush last
    if current_q is not None:
        flush(acc, current_q, current_ans)

    return acc


def main():
    parser = argparse.ArgumentParser(description="Parse qa.txt into JSON Lines.")
    parser.add_argument("--input", "-i", default="qa.txt", help="Path to input qa.txt")
    parser.add_argument(
        "--output",
        "-o",
        default=str(Path("data") / "qa.json"),
        help="Path to output JSON Lines file",
    )
    parser.add_argument(
        "--min-entity-len", type=int, default=3, help="Minimum length of entity tokens"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[parser] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    records = list(iter_records(lines))

    # Assign ids
    for idx, rec in enumerate(records, start=1):
        rec["id"] = idx

    # Write JSON Lines
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    by_type: Dict[str, int] = {}
    entity_total = 0
    for rec in records:
        by_type[rec["doc_type"]] = by_type.get(rec["doc_type"], 0) + 1
        entity_total += len(rec["entities"])

    print(f"[parser] Wrote {len(records)} records to {out_path}")
    print(f"[parser] Doc types: {by_type}")
    print(f"[parser] Total entities: {entity_total}")


if __name__ == "__main__":
    main()
