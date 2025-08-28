from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .parser import (
    iter_records as parse_qa_txt,
    extract_entities,
    detect_doc_type,
)


def _split_text_to_qa(text: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return lines[0], ""
    return lines[0], "\n".join(lines[1:])


def _normalize_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Accept a variety of input shapes and normalize to our schema
    q = obj.get("question") or obj.get("q")
    a = obj.get("answer_text") or obj.get("answer") or obj.get("a")

    if not q and not a:
        # Fallback: collapse any 'text' field into Q/A
        t = obj.get("text")
        if isinstance(t, str) and t.strip():
            q, a = _split_text_to_qa(t)

    if not isinstance(q, str):
        q = ""
    if not isinstance(a, str):
        a = ""

    combined = (q + "\n" + a).strip()
    doc_type = obj.get("doc_type") or detect_doc_type(combined)

    ents = obj.get("entities")
    if not isinstance(ents, list) or not all(isinstance(e, str) for e in ents):
        ents = extract_entities(combined)

    tags = obj.get("tags")
    if not isinstance(tags, list):
        tags = []
    if doc_type not in tags:
        tags = list(tags) + [doc_type]
    if ents and "has_entities" not in tags:
        tags.append("has_entities")

    out: Dict[str, Any] = {
        "question": q,
        "answer_text": a,
        "tags": tags,
        "entities": ents,
        "doc_type": doc_type,
    }

    # Preserve id if provided and valid; uniqueness handled later
    pid = obj.get("id")
    if isinstance(pid, int) and pid > 0:
        out["id"] = pid

    return out


def _parse_json_array(arr: List[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for itm in arr:
        if isinstance(itm, dict):
            records.append(_normalize_record(itm))
        elif isinstance(itm, str) and itm.strip():
            q, a = _split_text_to_qa(itm)
            records.append(_normalize_record({"question": q, "answer_text": a}))
    return records


def _parse_json_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    # Try JSON array/object first
    if text.startswith("[") or text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return _parse_json_array(obj)
            if isinstance(obj, dict):
                if "records" in obj and isinstance(obj["records"], list):
                    return _parse_json_array(obj["records"])
                # Single object treated as one record
                return [_normalize_record(obj)]
        except Exception:
            # Fall back to JSONL if non-standard JSON
            pass

    # JSON Lines (one JSON object per line)
    recs: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            # If a line is not JSON, treat it as plain text item
            q, a = _split_text_to_qa(s)
            recs.append(_normalize_record({"question": q, "answer_text": a}))
            continue
        if isinstance(obj, dict):
            recs.append(_normalize_record(obj))
        elif isinstance(obj, list):
            recs.extend(_parse_json_array(obj))
        elif isinstance(obj, str) and obj.strip():
            q, a = _split_text_to_qa(obj)
            recs.append(_normalize_record({"question": q, "answer_text": a}))
    return recs


def _first_text(node: ET.Element, names: List[str]) -> Optional[str]:
    # Prefer direct children, then any descendant
    for name in names:
        child = node.find(name)
        if child is not None:
            txt = (child.text or "").strip()
            if txt:
                return txt
    for name in names:
        child = node.find(f".//{name}")
        if child is not None:
            txt = (child.text or "").strip()
            if txt:
                return txt
    return None


def _parse_xml_file(path: Path) -> List[Dict[str, Any]]:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        # Not a parseable XML; treat as plain text
        return _parse_txt_file(path)

    recs: List[Dict[str, Any]] = []

    # Try to interpret each direct child as a record container
    children = list(root)
    for item in children:
        q = _first_text(item, ["question", "q"])
        a = _first_text(item, ["answer_text", "answer", "a", "text"])
        if q or a:
            recs.append(
                _normalize_record({"question": q or "", "answer_text": a or ""})
            )

    # Fallback: attempt to build a single record from root
    if not recs:
        q = _first_text(root, ["question", "q"])
        a = _first_text(root, ["answer_text", "answer", "a", "text"])
        content = (root.text or "").strip()
        if not (q or a) and content:
            q, a = _split_text_to_qa(content)
        if q or a:
            recs.append(
                _normalize_record({"question": q or "", "answer_text": a or ""})
            )

    return recs


def _parse_txt_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    parsed = list(parse_qa_txt(lines))
    out: List[Dict[str, Any]] = []
    for obj in parsed:
        out.append(_normalize_record(obj))
    if out:
        return out

    # Fallback: treat entire file as one record (first non-empty line as question)
    q, a = _split_text_to_qa(text)
    if q or a:
        return [_normalize_record({"question": q, "answer_text": a})]
    return []


def _gather_paths(dir_path: Path) -> List[Path]:
    patterns = ["*.json", "*.jsonl", "*.ndjson", "*.txt", "*.xml"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(dir_path.rglob(pat))

    # Exclude noisy/derived output folders and hidden dirs
    def _skip(p: Path) -> bool:
        parts = [str(x) for x in p.parts]
        for part in parts:
            if part.startswith("."):
                return True
            lp = part.lower()
            if lp.startswith("output"):
                return True
        return False

    files = [p for p in files if p.is_file() and not _skip(p)]
    # Keep deterministic order
    files = sorted(set(files))
    return files


def load_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset records from:
    - JSON Lines (.json/.jsonl/.ndjson)
    - JSON array/object (.json)
    - Plain TXT (qa.txt style using parser; fallback single-record)
    - XML (heuristic: direct child elements with question/answer tags; fallback)
    Also accepts a directory path and will load all supported files recursively.
    Ensures unique integer 'id' across all normalized records.
    """
    records: List[Dict[str, Any]] = []

    if path.is_dir():
        file_paths = _gather_paths(path)
        for p in file_paths:
            # Propagate source format tagging from each file
            recs = load_records(p)
            records.extend(recs)
    else:
        suffix = path.suffix.lower()
        try:
            if suffix in {".json", ".jsonl", ".ndjson"}:
                recs = _parse_json_file(path)
                for r in recs:
                    r["source_format"] = "json"
                records.extend(recs)
            elif suffix == ".xml":
                recs = _parse_xml_file(path)
                for r in recs:
                    r["source_format"] = "xml"
                records.extend(recs)
            elif suffix == ".txt":
                recs = _parse_txt_file(path)
                for r in recs:
                    r["source_format"] = "txt"
                records.extend(recs)
            else:
                # Unknown extension: attempt JSON, then XML, then TXT
                try:
                    recs = _parse_json_file(path)
                    for r in recs:
                        r["source_format"] = "json"
                    records.extend(recs)
                except Exception:
                    try:
                        recs = _parse_xml_file(path)
                        for r in recs:
                            r["source_format"] = "xml"
                        records.extend(recs)
                    except Exception:
                        recs = _parse_txt_file(path)
                        for r in recs:
                            r["source_format"] = "txt"
                        records.extend(recs)
        except Exception:
            # As a last resort, try plain text
            recs = _parse_txt_file(path)
            for r in recs:
                r["source_format"] = "txt"
            records.extend(recs)

    # Assign unique ids if missing or duplicated
    used_ids: Set[int] = set()
    next_id = 1
    for rec in records:
        rid = rec.get("id")
        if not isinstance(rid, int) or rid <= 0 or rid in used_ids:
            while next_id in used_ids:
                next_id += 1
            rec["id"] = next_id
            used_ids.add(next_id)
            next_id += 1
        else:
            used_ids.add(rid)

    return records
