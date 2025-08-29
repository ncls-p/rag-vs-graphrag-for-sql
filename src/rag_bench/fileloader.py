from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class FileItem:
    id: int
    source_path: str
    source_format: str  # one of: json, xml, txt
    content: str
    size_bytes: int


def _gather_paths(dir_path: Path) -> List[Path]:
    patterns = ["*.json", "*.jsonl", "*.ndjson", "*.txt", "*.xml"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(dir_path.rglob(pat))

    def _skip(p: Path) -> bool:
        for part in p.parts:
            if str(part).startswith("."):
                return True
        return False

    files = [p for p in files if p.is_file() and not _skip(p)]
    files = sorted(set(files))
    return files


def _infer_format(path: Path) -> Optional[str]:
    suf = path.suffix.lower()
    if suf in {".json", ".jsonl", ".ndjson"}:
        return "json"
    if suf == ".xml":
        return "xml"
    if suf == ".txt":
        return "txt"
    return None


def load_files(path: Path) -> List[FileItem]:
    items: List[FileItem] = []
    if path.is_dir():
        paths = _gather_paths(path)
    else:
        fm = _infer_format(path)
        paths = [path] if fm else []

    next_id = 1
    for p in paths:
        fm = _infer_format(p)
        if not fm:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        size = p.stat().st_size if p.exists() else 0
        items.append(
            FileItem(
                id=next_id,
                source_path=str(p),
                source_format=fm,
                content=text,
                size_bytes=int(size),
            )
        )
        next_id += 1
    return items

