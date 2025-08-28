from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ..bench.runner import run_benchmark, run_index
from ..config import Config
from ..io.neo4j import Neo4jIO
from ..io.qdrant import QdrantIO
from ..retrievals.neo4j import Neo4jRetriever
from ..retrievals.qdrant import QdrantRetriever

console = Console()


def _header(title: str) -> None:
    console.clear()
    console.print(
        Panel.fit(f"[bold cyan]rag-bench[/bold cyan] · {title}", border_style="cyan")
    )


def _error(msg: str) -> None:
    console.print(Panel(msg, title="Error", title_align="left", border_style="red"))


def _ok(msg: str) -> None:
    console.print(Panel(msg, title="OK", title_align="left", border_style="green"))


def _checkbox(
    title: str,
    options: Sequence[Tuple[str, str]],
    default_selected: Optional[Sequence[str]] = None,
) -> List[str]:
    default_set = set(default_selected or [])
    console.print(Panel.fit(title, style="cyan"))
    for i, (label, value) in enumerate(options, start=1):
        mark = "x" if value in default_set else " "
        console.print(f"[{i}] [{mark}] {label}")
    default_numbers = [
        str(i + 1) for i, (_, v) in enumerate(options) if v in default_set
    ]
    raw = Prompt.ask(
        "Enter comma-separated numbers",
        default=",".join(default_numbers) if default_numbers else "",
    )
    sel: List[str] = []
    if raw and raw.strip():
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for p in parts:
            try:
                idx = int(p)
            except Exception:
                continue
            if 1 <= idx <= len(options):
                sel.append(options[idx - 1][1])
    else:
        sel = list(default_set)
    return [s for s in sel if s in {v for _, v in options}]


def _pick_backends(defaults: Optional[List[str]] = None) -> List[str]:
    defaults = defaults or ["qdrant", "neo4j"]
    return _checkbox(
        "Select backends",
        options=[("Qdrant", "qdrant"), ("Neo4j", "neo4j")],
        default_selected=defaults,
    )


def _pick_format() -> Optional[List[str]]:
    console.print(Panel.fit("Restrict to format?", style="cyan"))
    options = [("All", None), ("JSON", "json"), ("TXT", "txt"), ("XML", "xml")]
    for i, (label, _) in enumerate(options, start=1):
        console.print(f"[{i}] {label}")
    sel = Prompt.ask("Choose 1", default="1")
    try:
        idx = int(sel)
    except Exception:
        idx = 1
    if not (1 <= idx <= len(options)):
        idx = 1
    value = options[idx - 1][1]
    return None if value is None else [value]


def _health() -> None:
    _header("Health Checks")
    cfg = Config()
    ok_oll, msg_oll = cfg.health_ollama()
    ok_qd, msg_qd = cfg.health_qdrant()
    ok_n4j, msg_n4j = cfg.health_neo4j()
    t = Table(title="Services", box=box.SIMPLE, show_lines=True)
    t.add_column("Service", style="bold")
    t.add_column("Status")
    t.add_row("Embeddings (Ollama)", ("✅ " if ok_oll else "❌ ") + msg_oll)
    t.add_row("Qdrant", ("✅ " if ok_qd else "❌ ") + msg_qd)
    t.add_row("Neo4j", ("✅ " if ok_n4j else "❌ ") + msg_n4j)
    console.print(t)
    Confirm.ask("Back to menu?", default=True)


def _index() -> None:
    _header("Index Knowledge Base")
    cfg = Config()
    data_path = Prompt.ask("KB path (folder)", default=str(Path("data")))
    if not data_path:
        return
    backends = _pick_backends()
    if not backends:
        _error("No backend selected")
        return
    console.print("[cyan]Indexing...[/cyan]")
    results: Dict[str, Any] = {}

    # Qdrant
    if "qdrant" in backends:
        import time

        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

        console.print(
            f"[bold cyan]Indexing Qdrant[/bold cyan] • URL: {cfg.qdrant_url} • Base collection: {QdrantIO(cfg).base_collection}"
        )
        qio = QdrantIO(cfg)
        files_task_id = None
        recs_task_id = None
        current_file = ""
        start_time = time.perf_counter()
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("• {task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:

            def _qcb(ev: Dict[str, Any]) -> None:
                nonlocal files_task_id, recs_task_id, current_file
                now = time.perf_counter()
                total_files = int(ev.get("total_files") or 1)
                total_records = int(ev.get("total_records") or 0)
                file_idx = int(ev.get("file_index") or 0)
                rec_idx = int(ev.get("record_index") or 0)
                file_path = ev.get("file_path") or ""
                if files_task_id is None:
                    files_task_id = progress.add_task("Files", total=total_files)
                if recs_task_id is None:
                    recs_task_id = progress.add_task("Records", total=total_records)
                # advance to current
                progress.update(files_task_id, completed=file_idx)
                progress.update(recs_task_id, completed=rec_idx)
                if file_path and file_path != current_file:
                    current_file = str(file_path)
                # throughput
                elapsed = max(now - start_time, 1e-6)
                fps = (file_idx / elapsed) if file_idx > 0 else 0.0
                avg_ms = (elapsed / max(file_idx, 1)) * 1000.0
                fdesc = f"Files ({file_idx}/{total_files}) • {fps:.2f} f/s • avg {avg_ms:.0f} ms"
                rdesc = (
                    f"Records • {Path(current_file).name}"
                    if current_file
                    else "Records"
                )
                progress.update(files_task_id, description=fdesc)
                progress.update(recs_task_id, description=rdesc)

            try:
                res = qio.index_dataset(Path(data_path), progress=_qcb)
                results["qdrant"] = res
            except Exception as e:
                _error(f"Qdrant index failed: {e}")
                Confirm.ask("Back to menu?", default=True)
                return

    # Neo4j
    if "neo4j" in backends:
        import time

        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

        console.print(f"[bold cyan]Indexing Neo4j[/bold cyan] • URI: {cfg.neo4j_uri}")
        n4j = Neo4jIO(cfg, use_read_only=False)
        files_task_id = None
        recs_task_id = None
        current_file = ""
        start_time = time.perf_counter()
        try:
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TextColumn("• {task.completed}/{task.total}"),
                TimeElapsedColumn(),
                transient=True,
                console=console,
            ) as progress:

                def _ncb(ev: Dict[str, Any]) -> None:
                    nonlocal files_task_id, recs_task_id, current_file
                    now = time.perf_counter()
                    total_files = int(ev.get("total_files") or 1)
                    total_records = int(ev.get("total_records") or 0)
                    file_idx = int(ev.get("file_index") or 0)
                    rec_idx = int(ev.get("record_index") or 0)
                    file_path = ev.get("file_path") or ""
                    if files_task_id is None:
                        files_task_id = progress.add_task("Files", total=total_files)
                    if recs_task_id is None:
                        recs_task_id = progress.add_task("Records", total=total_records)
                    progress.update(files_task_id, completed=file_idx)
                    progress.update(recs_task_id, completed=rec_idx)
                    if file_path and file_path != current_file:
                        current_file = str(file_path)
                    elapsed = max(now - start_time, 1e-6)
                    fps = (file_idx / elapsed) if file_idx > 0 else 0.0
                    avg_ms = (elapsed / max(file_idx, 1)) * 1000.0
                    fdesc = f"Files ({file_idx}/{total_files}) • {fps:.2f} f/s • avg {avg_ms:.0f} ms"
                    rdesc = (
                        f"Records • {Path(current_file).name}"
                        if current_file
                        else "Records"
                    )
                    progress.update(files_task_id, description=fdesc)
                    progress.update(recs_task_id, description=rdesc)

                stats = n4j.ingest(Path(data_path), progress=_ncb)
                results["neo4j"] = {
                    "documents": stats.documents,
                    "entities": stats.entities,
                    "mentions": stats.mentions,
                    "refers_to": stats.refers_to,
                }
        except Exception as e:
            _error(f"Neo4j ingest failed: {e}")
            Confirm.ask("Back to menu?", default=True)
            return
        finally:
            try:
                n4j.close()
            except Exception:
                pass

    # Summarize
    t = Table(title="Index Results", box=box.SIMPLE, show_lines=True)
    t.add_column("Backend", style="bold")
    t.add_column("Details")
    for bk, info in results.items():
        pretty = json.dumps(info, ensure_ascii=False)
        t.add_row(bk, pretty)
    console.print(t)
    Confirm.ask("Back to menu?", default=True)


def _search() -> None:
    _header("Search")
    console.print(Panel.fit("Backend", style="cyan"))
    console.print("[1] qdrant\n[2] neo4j")
    bsel = Prompt.ask("Choose 1", default="1")
    backend = "qdrant" if str(bsel).strip() != "2" else "neo4j"
    if not backend:
        return
    query = Prompt.ask("Query text")
    if not query:
        return
    cfg = Config()
    try:
        top_k_str = Prompt.ask("Top-K", default=str(cfg.top_k))
        top_k = int(top_k_str) if top_k_str else cfg.top_k
    except Exception:
        top_k = cfg.top_k
    fmts = _pick_format()

    retr_obj: Optional[Any] = None
    try:
        if backend == "qdrant":
            retr_obj = QdrantRetriever(cfg)
            hits = retr_obj.search(query=query, top_k=top_k, source_formats=fmts)
        else:
            retr_obj = Neo4jRetriever(cfg)
            hits = retr_obj.search(query=query, top_k=top_k, source_formats=fmts)
    except Exception as e:
        _error(str(e))
        Confirm.ask("Back to menu?", default=True)
        return
    finally:
        if backend == "neo4j" and retr_obj is not None:
            try:
                retr_obj.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    if not hits:
        _ok("No results")
        Confirm.ask("Back to menu?", default=True)
        return

    t = Table(title=f"Results ({backend})", box=box.MINIMAL_DOUBLE_HEAD)
    t.add_column("#", justify="right")
    t.add_column("ID", justify="right")
    t.add_column("Score")
    t.add_column("Doc Type")
    t.add_column("Entities")
    if backend == "neo4j":
        t.add_column("semantic")
        t.add_column("ent_jacc")
        t.add_column("neighbor")

    for i, h in enumerate(hits, start=1):
        pid = str(h.id)
        score = f"{h.score:.4f}"
        doc_type = (h.payload or {}).get("doc_type") if h.payload else ""
        ents = ", ".join((h.payload or {}).get("entities", [])[:5]) if h.payload else ""
        if backend == "neo4j":
            sem = f"{h.components.get('semantic', 0.0):.3f}"
            jac = f"{h.components.get('entity_jaccard', 0.0):.3f}"
            nei = f"{h.components.get('neighbor_boost', 0.0):.3f}"
            t.add_row(str(i), pid, score, str(doc_type or ""), ents, sem, jac, nei)
        else:
            t.add_row(str(i), pid, score, str(doc_type or ""), ents)
    console.print(t)

    # Optional details
    if Confirm.ask("Show first result payload?", default=False):
        p = hits[0].payload or {}
        console.print(
            Panel.fit(json.dumps(p, ensure_ascii=False, indent=2), title="Payload")
        )
    Confirm.ask("Back to menu?", default=True)


def _stats() -> None:
    _header("Neo4j Stats")
    cfg = Config()
    n4j = Neo4jIO(cfg, use_read_only=True)
    try:
        s = n4j.stats()
    except Exception as e:
        _error(str(e))
        return
    finally:
        try:
            n4j.close()
        except Exception:
            pass
    t = Table(title="Counts", box=box.SIMPLE)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in s.items():
        t.add_row(k, str(v))
    console.print(t)
    Confirm.ask("Back to menu?", default=True)


def _drop_neo4j() -> None:
    _header("Danger Zone · Drop Neo4j Graph")
    if not Confirm.ask(
        "This deletes all nodes/relationships. Continue?", default=False
    ):
        return
    txt = Prompt.ask("Type DROP to confirm")
    if txt != "DROP":
        _error("Confirmation mismatch; aborting")
        return
    cfg = Config()
    if not cfg.allow_destructive_ops:
        _error("Set ALLOW_DESTRUCTIVE_OPS=true to enable destructive operations")
        Confirm.ask("Back to menu?", default=True)
        return
    n4j = Neo4jIO(cfg, use_read_only=False)
    try:
        res = n4j.drop_graph()
        _ok(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        _error(str(e))
    finally:
        try:
            n4j.close()
        except Exception:
            pass
    Confirm.ask("Back to menu?", default=True)


def _drop_qdrant() -> None:
    _header("Danger Zone · Drop Qdrant Collections")
    if not Confirm.ask("This deletes Qdrant collections. Continue?", default=False):
        return
    txt = Prompt.ask("Type DROP to confirm")
    if txt != "DROP":
        _error("Confirmation mismatch; aborting")
        return
    cfg = Config()
    if not cfg.allow_destructive_ops:
        _error("Set ALLOW_DESTRUCTIVE_OPS=true to enable destructive operations")
        Confirm.ask("Back to menu?", default=True)
        return
    qio = QdrantIO(cfg)
    try:
        cols = qio.client.get_collections()
        names = [c.name for c in getattr(cols, "collections", [])]
        base = qio.base_collection
        targets = [n for n in names if (n == base or n.startswith(base + "_"))]
        if not targets:
            _ok("No matching collections to delete")
            Confirm.ask("Back to menu?", default=True)
            return
        deleted = []
        errors = []
        for n in targets:
            try:
                qio.client.delete_collection(n)
                deleted.append(n)
            except Exception as e:
                errors.append({"collection": n, "error": str(e)})
        msg = {
            "deleted": deleted,
            "errors": errors,
        }
        _ok(json.dumps(msg, ensure_ascii=False))
    except Exception as e:
        _error(str(e))
    Confirm.ask("Back to menu?", default=True)


def _drop_all() -> None:
    _header("Danger Zone · Drop ALL Data (Neo4j + Qdrant)")
    if not Confirm.ask(
        "This deletes ALL data in both backends. Continue?", default=False
    ):
        return
    txt = Prompt.ask("Type DROP ALL to confirm")
    if txt != "DROP ALL":
        _error("Confirmation mismatch; aborting")
        return
    cfg = Config()
    if not cfg.allow_destructive_ops:
        _error("Set ALLOW_DESTRUCTIVE_OPS=true to enable destructive operations")
        Confirm.ask("Back to menu?", default=True)
        return
    # Qdrant
    qio = QdrantIO(cfg)
    q_deleted = []
    q_errors = []
    try:
        cols = qio.client.get_collections()
        names = [c.name for c in getattr(cols, "collections", [])]
        base = qio.base_collection
        targets = [n for n in names if (n == base or n.startswith(base + "_"))]
        for n in targets:
            try:
                qio.client.delete_collection(n)
                q_deleted.append(n)
            except Exception as e:
                q_errors.append({"collection": n, "error": str(e)})
    except Exception as e:
        q_errors.append({"stage": "list", "error": str(e)})
    # Neo4j
    n4j = Neo4jIO(cfg, use_read_only=False)
    n_res = {}
    try:
        n_res = n4j.drop_graph()
    except Exception as e:
        n_res = {"error": str(e)}
    finally:
        try:
            n4j.close()
        except Exception:
            pass
    summary = {"qdrant": {"deleted": q_deleted, "errors": q_errors}, "neo4j": n_res}
    _ok(json.dumps(summary, ensure_ascii=False))
    Confirm.ask("Back to menu?", default=True)


def _benchmark() -> None:
    _header("Benchmark")
    backends = _pick_backends()
    if not backends:
        _error("No backend selected")
        return
    kb_path = Prompt.ask("KB path for indexing", default=str(Path("data")))
    if not kb_path:
        return
    qa_path = Prompt.ask("QA file for run", default=str(Path("data") / "qa.json"))
    if not qa_path:
        return
    cfg = Config()
    top_k = cfg.top_k
    try:
        tk = Prompt.ask("Top-K (run)", default=str(cfg.top_k))
        if tk:
            top_k = int(tk)
    except Exception:
        pass
    formats = _checkbox(
        "Restrict formats (for run)",
        options=[("json", "json"), ("txt", "txt"), ("xml", "xml")],
        default_selected=[],
    )

    # Index
    console.print("[cyan]Indexing KB...[/cyan]")
    try:
        run_index(backends, Path(kb_path))
    except Exception as e:
        _error(f"Index failed: {e}")
        Confirm.ask("Back to menu?", default=True)
        return

    # Run
    console.print("[cyan]Running benchmark...[/cyan]")
    try:
        res = run_benchmark(
            backends,
            Path(qa_path),
            Path("output") / "benchmark_results.json",
            top_k_override=top_k,
            formats=(formats or None),
        )
    except Exception as e:
        _error(f"Run failed: {e}")
        Confirm.ask("Back to menu?", default=True)
        return

    # Present summary
    console.print(Panel.fit("Completed", border_style="green"))
    console.print(
        Panel.fit(
            json.dumps(res.get("summary", {}), ensure_ascii=False, indent=2),
            title="Summary",
        )
    )
    if Confirm.ask("Save full result to file?", default=True):
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "benchmark_full.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        _ok(f"Wrote {out_path}")
    Confirm.ask("Back to menu?", default=True)


def main(argv: Optional[List[str]] = None) -> None:
    while True:
        _header("Interactive Console")
        console.print("[bold]What do you want to do?[/bold]")
        options = [
            ("Health Check", "health"),
            ("Index Knowledge Base", "index"),
            ("Search", "search"),
            ("Neo4j Stats", "stats"),
            ("Benchmark", "bench"),
            ("Danger: Drop Qdrant Collections", "drop_qdrant"),
            ("Danger: Drop Neo4j Graph", "drop_neo4j"),
            ("Danger: Drop ALL Data", "drop_all"),
            ("Exit", "exit"),
        ]
        for i, (label, _) in enumerate(options, start=1):
            console.print(f"[{i}] {label}")
        sel = Prompt.ask("Choose 1", default="1")
        try:
            idx = int(sel)
        except Exception:
            idx = 1
        if not (1 <= idx <= len(options)):
            idx = 1
        choice = options[idx - 1][1]

        if choice == "health":
            _health()
        elif choice == "index":
            _index()
        elif choice == "search":
            _search()
        elif choice == "stats":
            _stats()
        elif choice == "bench":
            _benchmark()
        elif choice == "drop_qdrant":
            _drop_qdrant()
        elif choice == "drop_neo4j":
            _drop_neo4j()
        elif choice == "drop_all":
            _drop_all()
        else:
            break


if __name__ == "__main__":
    main()
